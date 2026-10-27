import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from analyze import check_requirement
from rag import PolicyIndex

STATUS_COLORS = {
    "Addressed": "#e6f4ea",
    "Partially addressed": "#fef7e0",
    "Not addressed": "#fce8e6",
    "No evidence found": "#eeeeee",
    "Needs review": "#eeeeee",
}

DEFAULT_CHECKLIST = """Does the policy define who can access sensitive data?
Does it specify how long personal data can be retained?
Does it describe the procedure for reporting security incidents?
Does it explain how data is deleted when no longer needed?"""


def read_policy(file):
    if file.name.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    return file.read().decode("utf-8", errors="ignore")


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
    policy = read_policy(file) if file else pasted
    requirements = [line.strip("-• ").strip() for line in checklist.splitlines() if line.strip()]

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Add it to a .env file.")
    elif not policy.strip():
        st.error("Provide a policy document (upload or paste). If it's a scanned PDF, no text could be extracted.")
    elif not requirements:
        st.error("Add at least one requirement.")
    else:
        try:
            with st.spinner("Indexing policy..."):
                index = PolicyIndex(policy)
            progress = st.progress(0.0, text="Checking requirements...")
            rows = []
            for i, req in enumerate(requirements):
                rows.append(check_requirement(index, req))
                progress.progress((i + 1) / len(requirements), text=f"Checked {i + 1}/{len(requirements)}")
            progress.empty()
            st.session_state.report = pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Analysis failed: {e}")

if "report" in st.session_state:
    df = st.session_state.report
    st.divider()
    st.subheader("Compliance report")

    counts = df["Status"].value_counts()
    for col, status in zip(st.columns(len(STATUS_COLORS)), STATUS_COLORS):
        col.metric(status, int(counts.get(status, 0)))

    styled = df.style.hide(axis="index").apply(
        lambda row: [f"background-color: {STATUS_COLORS.get(row['Status'], '')}"] * len(row), axis=1
    ).format({"Retrieval score": "{:.2f}"})
    st.table(styled)

    st.download_button("Download CSV", df.to_csv(index=False), "compliance_report.csv", "text/csv")
