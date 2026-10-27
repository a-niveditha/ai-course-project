"""Judge one requirement with a small tool-calling agent.

The model gets the first retrieved passages up front. If they settle the question it submits a verdict
straight away (1 API call); otherwise it can search again with other wording or exact keywords first."""
import os
import re

from google.genai import types

from rag import MIN_SCORE, get_client

# Flash models allow only 20 requests/day on the free tier; flash-lite allows far more.
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_SEARCHES = 3
STATUSES = ["Addressed", "Partially addressed", "Not addressed"]

PROMPT = """You are a compliance analyst. Decide whether the policy addresses the requirement.
Base your verdict ONLY on policy text you have seen. Do not assume anything that is not written.

Requirement: {requirement}

Passages retrieved for this requirement:
{excerpts}

If these passages clearly settle it, call submit_verdict right away.
Otherwise, search first: policies often use different words (e.g. "breach" for "incident", "purge" for "delete").
- search_policy: meaning-based search with your own wording
- find_keyword: exact word or phrase search
Before concluding "Not addressed", try at least one search with different wording. You have at most {max_searches} searches.

Status meanings:
- "Addressed": the policy clearly and specifically covers the requirement
- "Partially addressed": the topic is mentioned or implied, but details are missing or vague
- "Not addressed": nothing in the policy relates to the requirement's topic
"""

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_policy",
        description="Semantic search over the policy. Returns the 3 most relevant passages.",
        parameters_json_schema={"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in your own words"}},
            "required": ["query"]},
    ),
    types.FunctionDeclaration(
        name="find_keyword",
        description="Exact, case-insensitive search for a word or short phrase. Returns surrounding text.",
        parameters_json_schema={"type": "object", "properties": {
            "term": {"type": "string", "description": "A single word or short phrase, e.g. 'breach'"}},
            "required": ["term"]},
    ),
    types.FunctionDeclaration(
        name="submit_verdict",
        description="Submit the final verdict for the requirement. Call this exactly once, last.",
        parameters_json_schema={"type": "object", "properties": {
            "reasoning": {"type": "string", "description": "One sentence on what the policy says about this topic, if anything"},
            "status": {"type": "string", "enum": STATUSES},
            "evidence": {"type": "string", "description": "The most relevant sentence copied word for word from the policy text you saw; empty if Not addressed"},
            "recommendation": {"type": "string", "description": "One short, concrete suggestion to improve the policy"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["reasoning", "status", "evidence", "recommendation", "confidence"]},
    ),
])


def run_agent(requirement, hits, index):
    """Returns (verdict, searches tried, all policy text the model saw, best search score, keyword hit count)."""
    seen = [chunk for chunk, _ in hits]
    searches, best, keyword_hits = [], hits[0][1], 0
    excerpts = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(seen))
    contents = [types.Content(role="user", parts=[types.Part(text=PROMPT.format(
        requirement=requirement, excerpts=excerpts, max_searches=MAX_SEARCHES))])]

    while True:
        # Must call a tool every turn; once the search budget is spent, only submit_verdict is allowed.
        allowed = ["submit_verdict"] if len(searches) >= MAX_SEARCHES else None
        res = get_client().models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[TOOLS],
                temperature=0,
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=allowed)),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if not res.function_calls:
            raise RuntimeError("Model did not call a tool")
        contents.append(res.candidates[0].content)

        replies = []
        for call in res.function_calls or []:
            args = dict(call.args or {})
            if call.name == "submit_verdict":
                return args, searches, seen, best, keyword_hits
            if call.name == "search_policy":
                found = index.search([args.get("query", "")])[0]
                searches.append(f'search "{args.get("query", "")}"')
                best = max(best, found[0][1])
                passages = [chunk for chunk, _ in found]
            else:
                passages = index.keyword_search(args.get("term", ""))
                searches.append(f'keyword "{args.get("term", "")}" ({len(passages)} hits)')
                keyword_hits += len(passages)
            seen += passages
            result = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "No matches."
            replies.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=replies))


def normalize(s):
    return re.sub(r"\s+", " ", s).strip(" \"'.…").lower()


def quote_found(evidence, passages):
    """Guards against hallucinated evidence: the quote must exist in policy text the model was shown."""
    q = normalize(evidence)
    return bool(q) and any(q in normalize(p) for p in passages)


def check_requirement(requirement, hits, index):
    verdict, searches, seen, best, keyword_hits = run_agent(requirement, hits, index)
    status = verdict.get("status", "")
    evidence = "" if status == "Not addressed" else verdict.get("evidence", "").strip()
    row = {"Requirement": requirement, "Status": status, "Evidence": evidence or "—",
           "Recommendation": verdict.get("recommendation", ""),
           "Searches tried": " → ".join(searches) or "First results were enough",
           "Human review": "", "Retrieval score": round(best, 2)}

    if status not in STATUSES:
        reason = f"Unexpected model output: {status!r}"
    elif status == "Not addressed" and best < MIN_SCORE and keyword_hits == 0:
        # Every search came back empty: we can't tell "absent" from "missed", so a human must check.
        row.update(Status="No evidence found", Evidence="No relevant passage retrieved",
                   **{"Human review": "Nothing relevant found by any search"})
        return row
    elif verdict.get("confidence") == "low":
        reason = f"Model said '{status}' with low confidence"
    elif status != "Not addressed" and not quote_found(evidence, seen):
        reason = f"Model said '{status}' but its quote isn't in the policy"
    else:
        # The agent read relevant passages (and searched again before ruling it out), so "Not addressed" means absent.
        return row

    row.update(Status="Needs review", **{"Human review": reason})
    return row
