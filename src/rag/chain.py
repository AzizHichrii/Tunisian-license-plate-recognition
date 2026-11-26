"""Valley RAG chain: rewrite question -> retrieve -> answer with citations."""
import json
import re
import time

from src.rag.llm import generate
from src.rag.retriever import retrieve
import os

FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite")
TOP_K = 6
RRF_K = 60   # standard constant for Reciprocal Rank Fusion

REWRITE_PROMPT = """You prepare search queries for a database about cars and Tunisian road law.
The Tunisian law documents are in French; general car documents are in English.
Question: {question}

Return ONLY a JSON object, no other text:
{{"language": "<ar|fr|en: language of the question>",
  "fr": "<the question rewritten as a French search query, using Tunisian legal vocabulary>",
  "en": "<the question rewritten as an English search query, using technical car vocabulary>"}}"""

SYSTEM_PROMPT = """You are Valley, an expert assistant about cars and Tunisian road law (code de la route).
Rules:
1. Answer ONLY with information from the CONTEXT. If the context does not contain the answer, say clearly
   that you don't have this information. Never invent articles, amounts or rules.
2. Answer in the SAME language as the user's question (Arabic, French or English).
3. Cite your sources with [1], [2]... matching the context numbers.
4. For legal questions (laws, fines, licences, obligations): name the text and the article, prefer the most
   recent text, mention its date, and end with one short line saying this is information, not legal advice.
   Do NOT add this line for technical or general car questions.
5. If the question is not about cars, driving or road law, politely say you only handle those topics.
Be clear and concise."""


def letters_ratio(text: str) -> float:
    s = re.sub(r"\s", "", text)
    return sum(ch.isalpha() for ch in s) / max(1, len(s))


def rewrite(question: str) -> dict:
    """Ask Gemini for search queries. If anything fails, fall back to the raw question."""
    try:
        raw, _ = generate(REWRITE_PROMPT.format(question=question), temperature=0, model=FAST_MODEL)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return {"language": data.get("language", "en"), "fr": data.get("fr", ""), "en": data.get("en", "")}
    except Exception:
        return {"language": "unknown", "fr": "", "en": ""}


def search(question: str, queries: dict) -> list[dict]:
    """Search with each query version, then merge the ranked lists with Reciprocal Rank Fusion."""
    fused, best = {}, {}
    for q in dict.fromkeys([question, queries["fr"], queries["en"]]):   # unique, keeps order
        if not q:
            continue
        ranked = [r for r in retrieve(q, k=TOP_K * 2) if letters_ratio(r["text"]) >= 0.6]
        for rank, r in enumerate(ranked):
            fused[r["id"]] = fused.get(r["id"], 0) + 1 / (RRF_K + rank + 1)
            if r["id"] not in best or r["score"] > best[r["id"]]["score"]:
                best[r["id"]] = r
    top_ids = sorted(fused, key=fused.get, reverse=True)[:TOP_K]
    return [best[i] for i in top_ids]

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = " | ".join(str(x) for x in (c["title"], c["section"], f"page {c['page']}" if c["page"] else "",
                                           f"date {c['doc_date']}") if x)
        parts.append(f"[{i}] ({meta})\n{c['text']}")
    return "\n\n".join(parts)


def ask(question: str, answer_model: str | None = None) -> dict:
    t0 = time.perf_counter()
    queries = rewrite(question)
    t1 = time.perf_counter()
    chunks = search(question, queries)
    t2 = time.perf_counter()
    prompt = f"CONTEXT:\n{build_context(chunks)}\n\nQUESTION: {question}"
    answer, model = generate(prompt, system=SYSTEM_PROMPT, model=answer_model)
    t3 = time.perf_counter()
    return {
        "answer": answer,
        "language": queries["language"],
        "queries": queries,
        "model": model,
        "sources": [{"n": i, "title": c["title"], "section": c["section"], "page": c["page"],
                     "source": c["source"], "date": str(c["doc_date"]), "score": round(c["score"], 3)}
                    for i, c in enumerate(chunks, 1)],
        "timings_ms": {"rewrite": round((t1 - t0) * 1000), "search": round((t2 - t1) * 1000),
                       "generate": round((t3 - t2) * 1000)},
    }


if __name__ == "__main__":
    for q in ["What is the fine for running a red light in Tunisia?",
              "Quelles sont les conditions pour obtenir un permis de conduire ?",
              "كم مرة يجب تغيير زيت المحرك؟"]:
        r = ask(q)
        print(f"\n{'=' * 80}\nQ: {q}\n[{r['model']}] lang={r['language']} | {r['timings_ms']}")
        print(f"queries: fr='{r['queries']['fr']}' | en='{r['queries']['en']}'")
        print(f"\n{r['answer']}\n\nSources:")
        for s in r["sources"]:
            print(f"  [{s['n']}] {s['score']} {s['title'][:40]} | {(s['section'] or '-')[:60]} | p.{s['page']}")