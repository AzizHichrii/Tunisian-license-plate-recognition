"""Compare answer models on the same questions: latency + answers for review."""
import time
from pathlib import Path

import mlflow
import pandas as pd

from src.detection.train import MLFLOW_URI, git_commit
from src.rag.chain import FAST_MODEL, ask

CANDIDATES = ["gemini-3.1-flash-lite"]
QUESTIONS = [
    ("en", "What is the fine for running a red light in Tunisia?"),
    ("en", "How does a turbocharger work?"),
    ("fr", "Quelles sont les conditions pour obtenir un permis de conduire ?"),
    ("fr", "Que signifie le feu rouge clignotant à un passage à niveau ?"),
    ("ar", "كم مرة يجب تغيير زيت المحرك؟"),
    ("ar", "ما هو نظام منع انغلاق المكابح؟"),
]
OUT = Path("runs/rag")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Valley_RAG")
    rows, report = [], ["# Valley benchmark\n"]

    for cand in CANDIDATES:
        with mlflow.start_run(run_name=f"answer_{cand}"):
            mlflow.log_params({"answer_model": cand, "rewrite_model": FAST_MODEL,
                               "top_k": 6, "git_commit": git_commit()})
            for lang, q in QUESTIONS:
                t0 = time.perf_counter()
                r = ask(q, answer_model=cand)
                total = time.perf_counter() - t0
                rows.append({"candidate": cand, "used": r["model"], "lang": lang, "question": q,
                             **r["timings_ms"], "total_s": round(total, 1),
                             "n_citations": sum(f"[{s['n']}]" in r["answer"] for s in r["sources"])})
                report.append(f"## [{cand} → used {r['model']}] ({lang}) {q}\n"
                              f"_{total:.1f}s_\n\n{r['answer']}\n")
                print(f"{cand:<22} {lang} {total:6.1f}s  {q[:50]}")
            df_c = pd.DataFrame([x for x in rows if x["candidate"] == cand])
            mlflow.log_metrics({"avg_total_s": df_c.total_s.mean(), "max_total_s": df_c.total_s.max(),
                                "avg_generate_ms": df_c.generate.mean(),
                                "avg_citations": df_c.n_citations.mean(),
                                "fallback_used": (df_c.used != cand).mean()})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bench.csv", index=False)
    (OUT / "bench_answers.md").write_text("\n".join(report), encoding="utf-8")
    print("\n", df.groupby("candidate")[["rewrite", "search", "generate", "total_s", "n_citations"]]
          .mean().round(1).to_string())
    print(f"\nAnswers saved to {OUT / 'bench_answers.md'} for quality review")


if __name__ == "__main__":
    main()