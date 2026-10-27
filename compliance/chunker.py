"""Chunking: one chunk per section of the policy, labelled with its heading path and pages."""
from compliance.config import MAX_WORDS, OVERLAP_WORDS


def fmt_pages(pages):
    return f"p. {pages[0]}" if pages[0] == pages[-1] else f"pp. {pages[0]}-{pages[-1]}"


def build_sections(lines):
    """Groups lines under their nearest heading. Titles keep the last two heading levels, e.g. 'GDPR > Your Rights'."""
    sections, stack = [], []
    for page, text, level in lines:
        if level and sections and not sections[-1]["words"] and stack and stack[-1][0] == level:
            stack[-1] = (level, f"{stack[-1][1]} {text}")  # heading wrapped onto a second line
            sections[-1]["title"] = " > ".join(t for _, t in stack[-2:])
        elif level:
            while stack and stack[-1][0] <= level:
                stack.pop()
            stack.append((level, text))
            sections.append({"title": " > ".join(t for _, t in stack[-2:]), "words": [], "pages": []})
        else:
            if not sections:
                sections.append({"title": "Introduction", "words": [], "pages": []})
            words = text.split()
            sections[-1]["words"] += words
            sections[-1]["pages"] += [page] * len(words)
    return sections


def make_chunks(sections):
    """Empty sections (a heading straight before a sub-heading) merge into the next one;
    long sections are split into overlapping pieces."""
    chunks, carry = [], None
    for s in sections:
        if carry:
            s = {"title": s["title"], "words": carry["words"] + s["words"], "pages": carry["pages"] + s["pages"]}
            carry = None
        if not s["words"]:
            carry = s
            continue
        step = MAX_WORDS - OVERLAP_WORDS
        for i in range(0, max(len(s["words"]) - OVERLAP_WORDS, 1), step):
            chunks.append({"title": s["title"], "text": " ".join(s["words"][i:i + MAX_WORDS]),
                           "pages": s["pages"][i:i + MAX_WORDS]})
    if carry and carry["words"]:
        chunks.append({"title": carry["title"], "text": " ".join(carry["words"]), "pages": carry["pages"]})
    return chunks
