"""OCR metrics: CER and exact match."""


def levenshtein(a: str, b: str) -> int:
    """Minimum number of edits (insert/delete/replace) to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds: list[str], targets: list[str]) -> float:
    """Character Error Rate: total edits / total characters (lower = better)."""
    edits = sum(levenshtein(p, t) for p, t in zip(preds, targets))
    return edits / sum(len(t) for t in targets)


def exact_match(preds: list[str], targets: list[str]) -> float:
    """Share of plates read 100% correctly (higher = better)."""
    return sum(p == t for p, t in zip(preds, targets)) / len(targets)