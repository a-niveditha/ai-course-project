"""Reading documents. Every reader returns [(page, line, heading_level)], where heading_level 0 = body text."""
from collections import Counter

import pymupdf


def read_pdf(data):
    """Headings are lines set in a larger font than the body text; bigger font = higher level."""
    lines = []
    for page_no, page in enumerate(pymupdf.open(stream=data, filetype="pdf"), 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).replace("​", "").strip()
                if text:
                    lines.append((page_no, text, round(max(s["size"] for s in line["spans"]))))
    if not lines:
        return []
    sizes = Counter()
    for _, text, size in lines:
        sizes[size] += len(text)
    body = sizes.most_common(1)[0][0]
    return [(p, text, size if size > body and len(text) < 100 else 0) for p, text, size in lines]


def read_text(text):
    """Plain text has no fonts: treat short lines without end punctuation (or markdown #) as headings."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hashes = len(line) - len(line.lstrip("#"))
        if hashes:
            level = 8 - hashes  # "#" outranks "##", and both outrank plain headings
        else:
            short = len(line.split()) <= 8 and (line[0].isupper() or line[0].isdigit())
            level = 1 if short and line[-1] not in ".,;:!?" else 0
        lines.append((1, line.lstrip("# "), level))
    return lines
