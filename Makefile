PY := .venv/bin/python
CLI := .venv/bin/nba-laya
GAMES := $(patsubst data/playbyplay_%.json,%,$(wildcard data/playbyplay_*.json))

export HF_HUB_DISABLE_XET = 1

.PHONY: setup serve rules laya scoreboard fetch

setup:
	uv venv -q -p 3.13 .venv
	uv pip install -q -p $(PY) -e . "laya[serve]"

## Run the Laya server in the foreground (own terminal tab).
serve:
	LAYA_MODELS=multilingual LAYA_PRELOAD=1 .venv/bin/laya-serve

## Rules baseline on every saved game.
rules:
	@for g in $(GAMES); do echo "== $$g"; $(CLI) replay --pbp data/playbyplay_$$g.json --out logs/rules_$$g.jsonl; done

## Zero-shot Laya on every saved game. Needs `make serve` running.
laya:
	@for g in $(GAMES); do echo "== $$g"; $(CLI) replay --pbp data/playbyplay_$$g.json --model laya --out logs/laya_$$g.jsonl; done

## Side-by-side scoreboard over everything in logs/.
scoreboard:
	$(CLI) scoreboard logs/*.jsonl

## Save a finished game's play-by-play: make fetch GAME=0022500001
fetch:
	$(CLI) fetch --game-id $(GAME)

## Pull a whole season (~35 min, resumable). SEASON is the two-digit start year.
## make season SEASON=25   -> data/season_25/ (2025-26 regular season + playoffs)
SEASON ?= 25
season:
	$(CLI) fetch-season --season $(SEASON) --playoffs

## Rules-grade every game in data/season_$(SEASON) -> logs/season_$(SEASON)/
grade:
	$(CLI) grade-dir data/season_$(SEASON) --out-dir logs/season_$(SEASON)

## Graded logs -> finetune/data/{train,calib,test}.jsonl, split by game.
export:
	$(CLI) export logs/season_*/*.jsonl --out-dir finetune/data

## season -> grade -> export in one go.
dataset: season grade export
