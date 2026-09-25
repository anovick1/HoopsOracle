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

## Rules-grade the training seasons in date order with pregame context.
## SEED seasons only feed the ratings (early-season priors); TRAIN seasons are graded.
SEED  ?= 23
TRAIN ?= 24 25
grade:
	$(CLI) grade-dir $(foreach s,$(TRAIN),data/season_$(s)) $(foreach s,$(SEED),--seed-dir data/season_$(s)) --out-dir logs

## Graded logs -> finetune/data/{train,calib,test}.jsonl, split by game date.
export:
	$(CLI) export $(foreach s,$(TRAIN),logs/season_$(s)/*.jsonl) --out-dir finetune/data

## Which snapshot fields carry signal (tree models, CPU, a few minutes).
ablation:
	$(PY) finetune/ablation.py $(foreach s,$(TRAIN),logs/season_$(s)) --max-rows 120000

## grade -> export. Run `make season SEASON=xx` for each season first.
dataset: grade export
