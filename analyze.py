"""Judge one requirement against the retrieved policy passages."""
import json
import re
import time

from google.genai import types

from rag import MIN_SCORE, get_client

LLM_MODEL = "gemini-3.8-flash"
STATUSES = ["Addressed", "Partially addressed", "Not addressed"]

PROMPT = """You are a compliance analyst. Decide whether the policy excerpts below address the requirement.
Use ONLY the excerpts. Do not assume anything that is not written.

Requirement: {requirement}

Policy excerpts:
{excerpts}

Return JSON with these keys:
- "status": "Addressed" (fully covered), "Partially addressed" (mentioned but incomplete or vague), or "Not addressed" (the excerpts do not cover it)
- "evidence": the most relevant sentence copied word for word from the excerpts; "" if Not addressed
- "recommendation": one short, concrete suggestion to improve the policy for this requirement
- "confidence": "high", "medium" or "low"
"""


def ask_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            res = get_client().models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            )
            return json.loads(res.text)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10)  # free tier rate limits


def normalize(s):
    return re.sub(r"\s+", " ", s).strip(" \"'.…").lower()


def quote_found(evidence, hits):
    """Guards against hallucinated evidence: the quote must exist in a retrieved chunk."""
    q = normalize(evidence)
    return bool(q) and any(q in normalize(chunk) for chunk, _ in hits)


def check_requirement(index, requirement):
    hits = index.search(requirement)
    best = hits[0][1]
    row = {"Requirement": requirement, "Status": "", "Evidence": "", "Recommendation": "",
           "Human review": "", "Retrieval score": round(best, 2)}

    # Retrieval found nothing relevant: we can't tell "absent" from "missed", so a human must check.
    if best < MIN_SCORE:
        row.update(Status="No evidence found",
                   Evidence="No relevant passage retrieved",
                   Recommendation="Confirm manually; if the policy doesn't cover this, add it.",
                   **{"Human review": "Nothing relevant retrieved"})
        return row

    excerpts = "\n\n".join(f"[{i + 1}] {chunk}" for i, (chunk, _) in enumerate(hits))
    verdict = ask_gemini(PROMPT.format(requirement=requirement, excerpts=excerpts))
    status = verdict.get("status", "")
    evidence = verdict.get("evidence", "").strip()
    row.update(Status=status, Evidence=evidence or "—", Recommendation=verdict.get("recommendation", ""))

    # Relevant passages were read, so "Not addressed" here means genuinely absent.
    if status not in STATUSES:
        reason = f"Unexpected model output: {status!r}"
    elif verdict.get("confidence") == "low":
        reason = f"Model said '{status}' with low confidence"
    elif status != "Not addressed" and not quote_found(evidence, hits):
        reason = f"Model said '{status}' but its quote isn't in the policy"
    else:
        return row

    row.update(Status="Needs review", **{"Human review": reason})
    return row
