"""Accuracy, Brier, and a 10-bin reliability curve."""

from __future__ import annotations


def probability_of_label(answer: dict, label) -> float | None:
    kind = answer.get("type")
    if kind == "noul":
        p_true = answer.get("noul")
        if "probabilities" in answer and "A" in answer["probabilities"]:
            p_true = answer["probabilities"]["A"]
        if p_true is None:
            return None
        return float(p_true) if label else 1.0 - float(p_true)
    if kind == "choice":
        probs = answer.get("probabilities") or {}
        if label not in probs:
            return None
        return float(probs[label])
    if kind == "score":
        probs = answer.get("probabilities") or {}
        key = str(label)
        if key not in probs:
            return None
        return float(probs[key])
    return None


def predicted_label(answer: dict):
    kind = answer.get("type")
    if kind == "noul":
        p = answer.get("noul", 0)
        if "probabilities" in answer and "A" in answer["probabilities"]:
            p = answer["probabilities"]["A"]
        return bool(p >= 0.5)
    if kind == "choice":
        return answer.get("choice")
    if kind == "score":
        probs = answer.get("probabilities") or {}
        if probs:
            return int(max(probs, key=lambda key: probs[key]))
        return int(round(answer.get("score", 0)))
    return None


def summarize(rows: list[dict], question: str) -> dict | None:
    pairs = []
    for row in rows:
        label = row.get("labels", {}).get(question)
        answer = row.get("answers", {}).get(question)
        if label is None or answer is None:
            continue
        p = probability_of_label(answer, label)
        if p is None:
            continue
        pairs.append((p, predicted_label(answer) == label))
    if not pairs:
        return None
    n = len(pairs)
    accuracy = sum(1 for _, hit in pairs if hit) / n
    brier = sum((1 - p) ** 2 for p, _ in pairs) / n
    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        group = [hit for p, hit in pairs if (lo <= p < hi) or (i == 9 and p == 1)]
        if not group:
            continue
        bins.append({
            "lo": lo,
            "hi": hi,
            "n": len(group),
            "hit_rate": sum(group) / len(group),
        })
    ece = 0.0
    for item in bins:
        mid = (item["lo"] + item["hi"]) / 2
        ece += item["n"] / n * abs(item["hit_rate"] - mid)
    return {"question": question, "n": n, "accuracy": accuracy, "brier": brier, "ece": ece, "bins": bins}


SCOREBOARD_QUESTIONS = ("winner", "possession_scores", "score_type", "run_continues", "shooter", "comeback")
