"""Tree models on the stat-line features, added one group at a time.

Answers two questions before any GPU time: which features carry signal, and
how far past the projection formula a well-fed model can get. The last column
is the number Laya has to reach.

    python finetune/statlines_ablation.py logs/statlines/statlines_23.jsonl logs/statlines/statlines_24.jsonl logs/statlines/statlines_25.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

TEST_FRAC = 0.15
GROUPS = ("projection", "tonight", "season", "form", "matchup", "teammates")


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    return rows


def _fg(s, i):
    try:
        return float(s.split("/")[i])
    except Exception:
        return 0.0


def features(r):
    s = r["state"]
    t = s["tonight"]
    score = list(s["score"].values())
    mine = s["score"].get(s["team"], 0)
    theirs = sum(score) - mine
    season, form, mu, proj = s.get("season", {}), s.get("form", {}), s.get("matchup", {}), s.get("projection", {})
    out = {
        "projection": {"proj_mean": proj.get("pts_mean", 0.0), "proj_sd": proj.get("pts_sd", 5.0)},
        "tonight": {
            "period": s["after_period"], "pts": t["pts"], "fgm": _fg(t["fg"], 0), "fga": _fg(t["fg"], 1),
            "tpm": _fg(t["3p"], 0), "tpa": _fg(t["3p"], 1), "ftm": _fg(t["ft"], 0), "fta": _fg(t["ft"], 1),
            "reb": t["reb"], "ast": t["ast"], "pf": t["pf"], "min": t["min"], "on_court": int(s.get("on_court_now", False)),
            "margin": mine - theirs, "abs_margin": abs(mine - theirs), "home": int(s.get("home", False)),
        },
        "season": {f"s_{k}": season.get(k, 0.0) for k in ("pts", "reb", "ast", "min", "fga", "pts_sd", "gp")},
        "form": {k: form.get(k, 0.0) for k in ("l5_pts", "l5_min", "l10_pts", "l10_min")} | {"min_pace": s.get("min_pace") or 1.0},
        "matchup": {"rest": mu.get("rest", 2), "b2b": int(mu.get("b2b", False)), "opp_def": mu.get("opp_def", 113), "opp_pace": mu.get("opp_pace", 99)},
        "teammates": {"vacancy": sum((s.get("teammates_out") or {}).values()), "n_out": len(s.get("teammates_out") or {})},
    }
    return out


def split(rows):
    games = sorted({(r["game_date"], r["game_id"]) for r in rows})
    cut = games[int(len(games) * (1 - TEST_FRAC))]
    return np.array([(r["game_date"], r["game_id"]) >= cut for r in rows])


def brier_rows(P, y):
    return float(np.mean((1 - P[np.arange(len(y)), y]) ** 2)), float(np.mean(P.argmax(1) == y))


def fit(X, y, train, test, n_classes):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0)
    clf.fit(X[train], y[train])
    P = clf.predict_proba(X[test])
    full = np.zeros((len(P), n_classes)); full[:, clf.classes_] = P
    return full


def formula_probs(rows, question):
    out = []
    for r in rows:
        a = r["answers"][question]
        if question == "pts_bucket":
            out.append([a["probabilities"][str(i)] for i in range(5)])
        else:
            p = a["noul"]
            out.append([1 - p, p])
    return np.array(out)


def run_player_question(rows, question, log=print):
    sel = [r for r in rows if r["kind"] == "player" and question in r["labels"]]
    y = np.array([int(r["labels"][question]) if question == "pts_bucket" else int(bool(r["labels"][question])) for r in sel])
    n_classes = 5 if question == "pts_bucket" else 2
    is_test = split(sel)
    train, test = ~is_test, is_test
    feats = [features(r) for r in sel]
    ck = np.array([r["checkpoint"] for r in sel])
    def report(name, P):
        b, a = brier_rows(P, y[test])
        per = " ".join(f"Q{c}:{brier_rows(P[ck[test] == c], y[test][ck[test] == c])[0]:.3f}" for c in (1, 2, 3))
        log(f"  {name:<22} brier {b:.4f}  acc {a:.3f}   {per}")
        return b
    log(f"\n== {question}  (train {train.sum()}, test {test.sum()})")
    base = report("formula", formula_probs(sel, question)[test])
    keys = []
    for g in GROUPS:
        keys += sorted(feats[0][g])
        X = np.array([[f[grp][k] for grp in GROUPS for k in sorted(f[grp]) if k in keys and k in f[grp]] for f in feats], dtype=float)
        b = report(f"+{g}", fit(X, y, train, test, n_classes))
    log(f"  skill of full tree over formula: {1 - b / base:+.1%}")


def run_top_scorer(rows, log=print):
    games = [r for r in rows if r["kind"] == "game"]
    is_test = split(games)
    # per-candidate rows
    X, y, gid, ck = [], [], [], []
    for i, r in enumerate(games):
        c = r["state"]["candidates"]
        best_proj = max(v["proj"] for v in c.values())
        for n, v in c.items():
            X.append([r["checkpoint"], v["pts"], _fg(v["fg"], 0), _fg(v["fg"], 1), v["min"], v["pf"], v.get("season_pts") or 0.0,
                      v["proj"], v["proj"] - best_proj, r["answers"]["top_scorer"]["probabilities"].get(n, 0.0)])
            y.append(int(r["labels"]["top_scorer"] == n)); gid.append(i); ck.append(r["checkpoint"])
    X, y, gid, ck = np.array(X, float), np.array(y), np.array(gid), np.array(ck)
    train_g = {i for i, t in enumerate(is_test) if not t}
    train = np.array([g in train_g for g in gid]); test = ~train
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
    clf.fit(X[train], y[train])
    p = clf.predict_proba(X[test])[:, 1]
    sums = defaultdict(float)
    for g, pi in zip(gid[test], p):
        sums[g] += pi
    pn = np.array([pi / sums[g] for g, pi in zip(gid[test], p)])
    def score(probs, mask):
        yy = y[test][mask]; pp = probs[mask]; gg = gid[test][mask]
        brier = float(np.mean((1 - pp[yy == 1]) ** 2))
        best = defaultdict(lambda: (-1, 0))
        for g, pi, yi in zip(gg, pp, yy):
            if pi > best[g][0]: best[g] = (pi, yi)
        return brier, float(np.mean([v[1] for v in best.values()]))
    f = X[test][:, -1]
    log(f"\n== top_scorer  (candidates train {train.sum()}, test {test.sum()})")
    for name, probs in (("formula (MC)", f), ("tree", pn)):
        b, a = score(probs, np.ones(len(probs), bool))
        per = " ".join(f"Q{c}:{score(probs, ck[test] == c)[1]:.3f}" for c in (1, 2, 3))
        log(f"  {name:<22} brier {b:.4f}  acc {a:.3f}   acc by ckpt {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--regular-only", action="store_true", default=True)
    args = ap.parse_args()
    rows = load(args.paths)
    if args.regular_only:
        rows = [r for r in rows if r["game_id"].startswith("002")]
    print(f"{len(rows)} rows, {len({r['game_id'] for r in rows})} games (regular season)")
    for q in ("pts_bucket", "thirty_plus", "triple_double"):
        run_player_question(rows, q)
    run_top_scorer(rows)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
