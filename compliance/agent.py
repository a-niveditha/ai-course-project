"""The tool-calling loop for one requirement.

The model gets the first retrieved passages up front. If they settle the question it submits a verdict
straight away (1 API call); otherwise it can search again with other wording or exact keywords first."""
from google.genai import types

from compliance.config import LLM_MODEL, MAX_SEARCHES
from compliance.llm import get_client
from compliance.tools import DECLARATIONS, run_tool

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


def numbered(passages):
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def run_agent(requirement, hits, index):
    """Returns the verdict plus a trace: searches tried, all policy text the model saw,
    the best semantic score and the number of keyword hits."""
    trace = {"searches": [], "seen": [chunk for chunk, _ in hits], "best": hits[0][1], "keyword_hits": 0}
    contents = [types.Content(role="user", parts=[types.Part(text=PROMPT.format(
        requirement=requirement, excerpts=numbered(trace["seen"]), max_searches=MAX_SEARCHES))])]

    while True:
        # Must call a tool every turn; once the search budget is spent, only submit_verdict is allowed.
        allowed = ["submit_verdict"] if len(trace["searches"]) >= MAX_SEARCHES else None
        res = get_client().models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[DECLARATIONS],
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
        for call in res.function_calls:
            args = dict(call.args or {})
            if call.name == "submit_verdict":
                return args, trace
            passages, log, score = run_tool(index, call.name, args)
            trace["searches"].append(log)
            trace["seen"] += passages
            if score is None:
                trace["keyword_hits"] += len(passages)
            else:
                trace["best"] = max(trace["best"], score)
            replies.append(types.Part.from_function_response(
                name=call.name, response={"result": numbered(passages) or "No matches."}))
        contents.append(types.Content(role="user", parts=replies))
