"""Embed all chunks and (re)build the pgvector table."""
import json
import time
from pathlib import Path

from src.rag.store import EMBED_DIM, EMBED_MODEL, embed, get_conn

CHUNKS = Path("data/rag/processed/chunks.jsonl")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS rag_chunks (
    id        TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    title     TEXT,
    section   TEXT,
    page      INT,
    category  TEXT,
    source    TEXT,
    language  TEXT,
    doc_date  DATE,
    file      TEXT,
    embedding vector({EMBED_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
"""

COLUMNS = ["id", "text", "title", "section", "page", "category", "source", "language", "doc_date", "file"]


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    # Give each chunk its context (title + section) before embedding it
    texts = [" — ".join(p for p in (c["title"], c["section"]) if p) + "\n" + c["text"] for c in chunks]

    t0 = time.perf_counter()
    vectors = embed(texts)
    print(f"Embedded {len(chunks)} chunks with {EMBED_MODEL} in {time.perf_counter() - t0:.0f}s "
          f"-> shape {vectors.shape}")

    rows = [(c["id"], c["text"], c["title"], c["section"], c["page"], c["category"], c["source"],
             c["language"], c["date"], c["file"], v) for c, v in zip(chunks, vectors)]
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.execute("TRUNCATE rag_chunks")   # full rebuild: re-running gives the same result
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO rag_chunks ({', '.join(COLUMNS)}, embedding) "
                f"VALUES ({', '.join(['%s'] * (len(COLUMNS) + 1))})", rows)
        count = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0]
    print(f"{count} rows in rag_chunks")


if __name__ == "__main__":
    main()