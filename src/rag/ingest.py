"""Extract text from RAG sources and split it into chunks (JSONL)."""
import csv
import json
import re
from pathlib import Path

import pymupdf

RAW = Path("data/rag/raw")
SOURCES = Path("data/rag/sources.csv")
OUT = Path("data/rag/processed/chunks.jsonl")
CHUNK_SIZE, OVERLAP = 1200, 200

ARTICLE_RE = re.compile(r"(?:Article|Art\.)\s+(\d+(?:\s*(?:bis|ter|quater))?)")
TEXT_RE = re.compile(
    r"^\s*((?:(?:Loi|Décret|Decret)[^\n]{0,20}n\s*°\s*[\d-]+|Arrêtés?\s+d[ue]s?\s+[Mm]inistres?)[^\n]{0,70})",
    re.M)
HEADER_RE = re.compile(r"Imprimerie Officielle de la République Tunisienne\s*\d*")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Cut into ~size chars, preferably at a paragraph or sentence end, with overlap."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if len(text) > 50 else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
        space = text.find(" ", start)
        if 0 <= space < end:
            start = space + 1   # never start a chunk in the middle of a word
    return [c for c in chunks if len(c) > 50]


def pdf_sections(path: Path):
    """Law PDFs: one section per 'Article N', labelled with the law/decree it belongs to."""
    full, offsets = "", []
    for i, page in enumerate(pymupdf.open(path)):
        offsets.append((len(full), i + 1))
        full += HEADER_RE.sub("", page.get_text()) + "\n"
    pos, current_text = 0, ""
    for part in re.split(r"(?=\n\s*(?:Article|Art\.)\s+\d+)", full):
        page = max(n for off, n in offsets if off <= pos)
        pos += len(part)
        m = ARTICLE_RE.match(part.strip())
        label = f"Article {m.group(1)}" if m else ""
        if current_text and label:
            label = f"{current_text} — {label}"
        yield label, page, part
        found = TEXT_RE.findall(part)   # a new law/decree starts inside this part
        if found:
            current_text = re.sub(r"\s+", " ", found[-1]).strip(" ,.")[:90]


def wiki_sections(path: Path):
    """Wikipedia JSON: one section per heading."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    parts = re.split(r"\n=+ (.+?) =+\n", "\n" + doc["text"])
    yield "Introduction", None, parts[0]
    for i in range(1, len(parts) - 1, 2):
        yield parts[i], None, parts[i + 1]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with SOURCES.open(encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    n, per_cat, lengths = 0, {}, []
    with OUT.open("w", encoding="utf-8") as out:
        for src in sources:
            path = RAW / src["file"]
            sections = pdf_sections(path) if path.suffix == ".pdf" else wiki_sections(path)
            for section, page, text in sections:
                for chunk in split_text(text):
                    rec = {"id": f"{path.stem}-{n}", "text": chunk, "title": src["title"],
                           "section": section, "page": page, "category": src["category"],
                           "source": src["source"], "language": src["language"],
                           "date": src["date"], "file": src["file"]}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    per_cat[src["category"]] = per_cat.get(src["category"], 0) + 1
                    lengths.append(len(chunk))

    print(f"{n} chunks from {len(sources)} documents -> {OUT}")
    print("per category:", per_cat)
    print(f"chunk length: avg {sum(lengths) // len(lengths)}, max {max(lengths)} chars")


if __name__ == "__main__":
    main()