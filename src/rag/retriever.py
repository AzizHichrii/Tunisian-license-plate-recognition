"""Find the chunks most similar to a question (cosine similarity in pgvector)."""
from src.rag.store import embed, get_conn

QUERY = """
SELECT id, text, title, section, page, category, source, doc_date,
       1 - (embedding <=> %(q)s) AS score
FROM rag_chunks
{where}
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


def retrieve(question: str, k: int = 5, category: str | None = None) -> list[dict]:
    q = embed([question])[0]
    where = "WHERE category = %(cat)s" if category else ""
    with get_conn() as conn:
        cur = conn.execute(QUERY.format(where=where), {"q": q, "k": k, "cat": category})
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


if __name__ == "__main__":
    questions = [
        "What is the fine for running a red light in Tunisia?",
        "Quelles sont les conditions pour obtenir un permis de conduire ?",
        "كم مرة يجب تغيير زيت المحرك؟",          # How often should engine oil be changed?
        "How does an anti-lock braking system work?",
        "What do the numbers on a Tunisian license plate mean?",
    ]
    for q in questions:
        print(f"\n### {q}")
        for r in retrieve(q, k=3):
            where = r["section"] or "-"
            page = f" p.{r['page']}" if r["page"] else ""
            print(f"  {r['score']:.3f} | {r['title'][:40]} | {where[:60]}{page}")
            print(f"        {r['text'][:120].replace(chr(10), ' ')}…")