"""Turns the agent's verdict into a report row, applying the human-review rules."""
from compliance.agent import run_agent
from compliance.config import MIN_SCORE, STATUSES
from compliance.index import normalize


def quote_found(evidence, passages):
    """Guards against hallucinated evidence: the quote must exist in policy text the model was shown."""
    q = normalize(evidence)
    return bool(q) and any(q in normalize(p) for p in passages)


def with_page(row, evidence, index):
    page = index.page_of(evidence) if evidence else ""
    if page:
        row["Evidence"] = f"{evidence} ({page})"
    return row


def check_requirement(requirement, hits, index):
    verdict, trace = run_agent(requirement, hits, index)
    status = verdict.get("status", "")
    evidence = "" if status == "Not addressed" else verdict.get("evidence", "").strip()
    row = {"Requirement": requirement, "Status": status, "Evidence": evidence or "—",
           "Recommendation": verdict.get("recommendation", ""),
           "Searches tried": " → ".join(trace["searches"]) or "First results were enough",
           "Human review": "", "Retrieval score": round(trace["best"], 2)}

    if status not in STATUSES:
        reason = f"Unexpected model output: {status!r}"
    elif status == "Not addressed" and trace["best"] < MIN_SCORE and trace["keyword_hits"] == 0:
        # Every search came back empty: we can't tell "absent" from "missed", so a human must check.
        row.update(Status="No evidence found", Evidence="No relevant passage retrieved",
                   **{"Human review": "Nothing relevant found by any search"})
        return row
    elif verdict.get("confidence") == "low":
        reason = f"Model said '{status}' with low confidence"
    elif status != "Not addressed" and not quote_found(evidence, trace["seen"]):
        reason = f"Model said '{status}' but its quote isn't in the policy"
    else:
        # The agent read relevant passages (and searched again before ruling it out), so "Not addressed" means absent.
        return with_page(row, evidence, index)

    row.update(Status="Needs review", **{"Human review": reason})
    return with_page(row, evidence, index)
