"""Retrieval over one policy: semantic search, keyword search and page lookup.
Everything lives in memory - one policy per run, so no vector database is needed."""
import re

from compliance.chunker import build_sections, fmt_pages, make_chunks
from compliance.config import TOP_K
from compliance.llm import embed


def normalize(s):
    return re.sub(r"\s+", " ", s).strip(" \"'.…").lower()


class PolicyIndex:
    def __init__(self, lines):
        self.chunks = make_chunks(build_sections(lines))
        self.page_text = {}
        for page, text, _ in lines:
            self.page_text[page] = self.page_text.get(page, "") + " " + text
        self.vectors = embed([f"{c['title']}\n{c['text']}" for c in self.chunks], "RETRIEVAL_DOCUMENT")

    def label(self, c):
        return f"[{c['title']}, {fmt_pages(c['pages'])}]\n{c['text']}"

    def search(self, queries, k=TOP_K):
        """Embeds all queries in one call. Returns, per query, [(labelled chunk, score), ...] best first."""
        results = []
        for q in embed(queries, "RETRIEVAL_QUERY"):
            scores = self.vectors @ q
            top = scores.argsort()[::-1][:k]
            results.append([(self.label(self.chunks[i]), float(scores[i])) for i in top])
        return results

    def keyword_search(self, term, limit=5, window=200):
        """Whole-word, case-insensitive match - runs locally, no API call. Returns snippets tagged with their page."""
        snippets, pattern = [], re.compile(rf"\b{re.escape(term)}", re.I)  # "dress" must not match "address"
        for page, text in self.page_text.items():
            last_end = -1
            for m in pattern.finditer(text):
                if len(snippets) >= limit:
                    return snippets
                if m.start() > last_end:  # skip matches already inside the previous snippet
                    last_end = m.end() + window
                    snippets.append(f"[p. {page}] {text[max(0, m.start() - window):last_end].strip()}")
        return snippets

    def page_of(self, quote):
        """Page where a quote appears; a range if it spans a page break."""
        q = normalize(quote)
        for page, text in self.page_text.items():
            if q in normalize(text):
                return f"p. {page}"
        for c in self.chunks:
            if q in normalize(c["text"]):
                return fmt_pages(c["pages"])
        return ""
