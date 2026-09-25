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

| id | type | asked when | graded against |
| --- | --- | --- | --- |
| `winner` | home/away | every possession | final score |
| `possession_scores` | yes/no | every possession | team with the ball scores before the ball changes hands |
| `score_type` | two/three/free_throw | every possession | kind of the next made basket, same period |
| `run_continues` | yes/no | a 6-0 or better run | run team outscores opponent over the next 3 possessions |
| `shooter` | one of five names | all five on the floor known (~80% of possessions) | player on the team with the ball who takes its next field-goal attempt |
| `comeback` | yes/no | once, when a team first falls 10 behind | that team leads at any later point |

Training uses every possession. What gets posted is decided separately in
`src/nba_laya/policy.py`: win-probability swings of 8+ points, runs of 8+,
`shooter` / `possession_scores` / `score_type` in the last five minutes of a
game within 8, and every `comeback` call.

Lineups are rebuilt each period from substitutions and from anyone who records
an action, and only trusted at exactly five, since between-period subs are not
always on the tape.

Rules baseline on the full 2025-26 season (1,315 games):

| question | n | accuracy | Brier |
| --- | --- | --- | --- |
| `winner` | 266,336 | 0.742 | 0.163 |
| `possession_scores` | 266,336 | 0.500 | 0.250 |
| `score_type` | 257,039 | 0.534 | 0.390 |
| `run_continues` | 34,874 | 0.666 | 0.222 |
| `shooter` | 197,262 | 0.258 | 0.621 |
| `comeback` | 2,200 | 0.635 | 0.215 |

## Environment notes

- `HF_HUB_DISABLE_XET=1` is set by the Makefile. Without it the Laya weights
  download from Hugging Face can stall at 0 bytes.
- Laya inference is ~400 ms per possession on Apple silicon CPU. No GPU needed
  for serving. Fine-tuning runs in Colab; see `finetune/`.
