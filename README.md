# NBA Laya

Watch NBA games at possession cadence, ask a decision model a fixed set of typed
questions, grade every answer against what actually happened, and keep score.
Laya (open weights, local) first. Jev (hosted) on the same request shape when a
key exists.

## Setup

```bash
make setup                # uv venv on Python 3.13, installs the package + laya[serve]
```

## Data

The play-by-play comes from the NBA's live CDN (`cdn.nba.com/static/json/liveData`).
It sits behind Akamai, which rejects plain Python and curl TLS handshakes. The
feed module uses `curl_cffi` to present a Chrome handshake; the nba.com Referer
header is also required. `nba_api` wraps the same URL with `requests` and is
blocked for the same reason.

```bash
make season SEASON=25     # 2025-26 regular season + playoffs -> data/season_25/  (~35 min, resumable)
make grade  SEASON=25     # rules baseline + truth labels     -> logs/season_25/
make export               # Laya training rows, split by game -> finetune/data/{train,calib,test}.jsonl
make dataset              # all three
```

Run these from your own terminal. The season pull is a long-lived process.

## Replay and scoreboard

```bash
make rules                # rules baseline on every game in data/*.json
make serve                # laya-serve, multilingual checkpoint, foreground (own tab)
make laya                 # zero-shot Laya on the same games
make scoreboard           # Laya vs rules: n, accuracy, Brier, ECE per question
```

## Questions

| id | type | graded against |
| --- | --- | --- |
| `timeout_next2` | yes/no | a timeout within the next 2 possession changes |
| `run_continues` | yes/no | run team outscores opponent over the next 3 possessions (6+ runs only) |
| `next_score` | home/away | team of the next scoring play, same period |
| `sub_next2` | yes/no | a player with 4+ fouls subbed out within 2 possessions |
| `winner` | home/away | final score |
| `swing` | 0-3 | change in a deterministic win-probability estimate (not on the scoreboard) |

## Environment notes

- `HF_HUB_DISABLE_XET=1` is set by the Makefile. Without it the Laya weights
  download from Hugging Face can stall at 0 bytes.
- Laya inference is ~400 ms per possession on Apple silicon CPU. No GPU needed
  for serving. Fine-tuning runs in Colab; see `finetune/`.
