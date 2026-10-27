import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from compliance.analyzer import check_requirement
from compliance.index import PolicyIndex
from compliance.loader import read_pdf, read_text

# (status text colour, translucent row tint) - readable on both light and dark themes
STATUS_COLORS = {
    "Addressed": ("#2e9e5b", "rgba(46, 158, 91, 0.10)"),
    "Partially addressed": ("#c98a12", "rgba(201, 138, 18, 0.10)"),
    "Not addressed": ("#d64545", "rgba(214, 69, 69, 0.10)"),
    "No evidence found": ("#8a8f98", "rgba(138, 143, 152, 0.14)"),
    "Needs review": ("#8a8f98", "rgba(138, 143, 152, 0.14)"),
}


def style_row(row):
    text, tint = STATUS_COLORS.get(row["Status"], ("", ""))
    return [f"background-color: {tint}; color: {text}; font-weight: 600" if col == "Status"
            else f"background-color: {tint}" for col in row.index]

DEFAULT_CHECKLIST = """Does the policy define who can access sensitive data?
Does it specify how long personal data can be retained?
Does it describe the procedure for reporting security incidents?
Does it explain how data is deleted when no longer needed?"""


def read_policy(file):
    if file.name.lower().endswith(".pdf"):
        return read_pdf(file.getvalue())
    return read_text(file.getvalue().decode("utf-8", errors="ignore"))


st.set_page_config(page_title="Policy Compliance Checker", layout="wide")
st.title("Policy Compliance Checker")
st.caption("Checks a policy document against a requirements checklist. Grey rows need human review.")

left, right = st.columns(2)
with left:
    st.subheader("Policy document")
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    pasted = st.text_area("Or paste the policy text", height=180)
with right:
    st.subheader("Requirements checklist")
    checklist = st.text_area("One requirement per line", value=DEFAULT_CHECKLIST, height=290)

if st.button("Generate report", type="primary"):
    policy = read_policy(file) if file else read_text(pasted)
    requirements = [line.strip("-• ").strip() for line in checklist.splitlines() if line.strip()]

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Add it to a .env file.")
    elif not policy:
        st.error("Provide a policy document (upload or paste). If it's a scanned PDF, no text could be extracted.")
    elif not requirements:
        st.error("Add at least one requirement.")
    else:
        try:
            with st.spinner("Indexing policy..."):
                index = PolicyIndex(policy)
                all_hits = index.search(requirements)
            progress = st.progress(0.0, text="Checking requirements...")
            rows = []
            for i, (req, hits) in enumerate(zip(requirements, all_hits)):
                rows.append(check_requirement(req, hits, index))
                progress.progress((i + 1) / len(requirements), text=f"Checked {i + 1}/{len(requirements)}")
            progress.empty()
            st.session_state.report = pd.DataFrame(rows)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                st.error("Gemini free-tier rate limit reached. Wait a minute and try again.")
            else:
                st.error(f"Analysis failed: {e}")

if "report" in st.session_state:
    df = st.session_state.report
    st.divider()
    st.subheader("Compliance report")

    counts = df["Status"].value_counts()
    for col, status in zip(st.columns(len(STATUS_COLORS)), STATUS_COLORS):
        col.metric(status, int(counts.get(status, 0)))

    styled = df.style.hide(axis="index").apply(style_row, axis=1).format({"Retrieval score": "{:.2f}"})
    st.table(styled)

    st.download_button("Download CSV", df.to_csv(index=False), "compliance_report.csv", "text/csv")
