"""The tools the agent can call: their declarations (what Gemini sees) and how each one runs."""
from google.genai import types

from compliance.config import STATUSES

DECLARATIONS = types.Tool(function_declarations=[
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


def run_tool(index, name, args):
    """Runs a search tool. Returns (passages, log entry for the report, best semantic score or None)."""
    if name == "search_policy":
        query = args.get("query", "")
        found = index.search([query])[0]
        return [chunk for chunk, _ in found], f'search "{query}"', found[0][1]
    term = args.get("term", "")
    passages = index.keyword_search(term)
    return passages, f'keyword "{term}" ({len(passages)} hits)', None
