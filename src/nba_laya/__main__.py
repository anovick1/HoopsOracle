"""CLI. Replay is the path that works before a live game exists."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nba_laya.client import SystemOneClient
from nba_laya.replay import format_scoreboard, load_payload, replay, write_log


def main() -> None:
    parser = argparse.ArgumentParser(prog="nba-laya")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("replay", help="grade one finished game")
    play.add_argument("--game-id", help="10-digit NBA game id, fetched from the live CDN")
    play.add_argument("--pbp", type=Path, help="play-by-play JSON file")
    play.add_argument("--model", default="rules", choices=("rules", "laya", "jev"))
    play.add_argument("--out", type=Path, help="JSONL log path")
    play.add_argument("--base-url", default=os.environ.get("LAYA_BASE_URL", "http://127.0.0.1:8000"))

    board = sub.add_parser("scoreboard", help="compare models over graded logs")
    board.add_argument("logs", nargs="+", type=Path)

    fetch = sub.add_parser("fetch", help="save a game's play-by-play under data/")
    fetch.add_argument("--game-id", required=True)
    fetch.add_argument("--out-dir", type=Path, default=Path("data"))

    season = sub.add_parser("fetch-season", help="save a whole season's play-by-play under data/")
    season.add_argument("--season", type=int, required=True, help="two-digit start year, 25 for 2025-26")
    season.add_argument("--playoffs", action="store_true", help="include playoff games")
    season.add_argument("--out-dir", type=Path, default=Path("data"))
    season.add_argument("--pause", type=float, default=0.6)

    exp = sub.add_parser("export", help="graded logs -> Laya training rows, split by game")
    exp.add_argument("logs", nargs="+", type=Path)
    exp.add_argument("--out-dir", type=Path, default=Path("finetune/data"))

    grade = sub.add_parser("grade-dir", help="rules-grade every play-by-play file in a directory")
    grade.add_argument("pbp_dir", type=Path)
    grade.add_argument("--out-dir", type=Path, default=Path("logs/graded"))

    args = parser.parse_args()
    if args.command == "export":
        from nba_laya.export import export

        print(export(args.logs, args.out_dir))
        return
    if args.command == "grade-dir":
        _grade_dir(args)
        return
    if args.command == "replay":
        _replay(args)
    elif args.command == "scoreboard":
        _scoreboard(args)
    elif args.command == "fetch":
        _fetch(args)
    elif args.command == "fetch-season":
        _fetch_season(args)


def _grade_dir(args: argparse.Namespace) -> None:
    import json

    files = sorted(args.pbp_dir.glob("playbyplay_*.json"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for path in files:
        out = args.out_dir / (path.stem.replace("playbyplay_", "rules_") + ".jsonl")
        if out.exists():
            continue
        try:
            rows = replay(json.loads(path.read_text()), model="rules")
        except Exception as exc:  # one bad tape should not stop the season
            failed += 1
            print(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        write_log(rows, out)
        done += 1
        if done % 100 == 0:
            print(f"graded {done}")
    print(f"graded {done}, failed {failed}, files {len(files)}")


def _fetch_season(args: argparse.Namespace) -> None:
    from nba_laya.bulk import fetch_many, playoff_ids, regular_season_ids

    ids = regular_season_ids(args.season)
    if args.playoffs:
        ids += playoff_ids(args.season)
    counts = fetch_many(ids, args.out_dir / f"season_{args.season:02d}", pause=args.pause)
    print(counts)


def _scoreboard(args: argparse.Namespace) -> None:
    import json

    from nba_laya.metrics import SCOREBOARD_QUESTIONS, summarize

    by_model: dict[str, list[dict]] = {}
    for path in args.logs:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                by_model.setdefault(row["model"], []).append(row)
    models = sorted(by_model)
    header = f"{'question':<16}" + "".join(f"{m:>26}" for m in models)
    print(header)
    print(f"{'':<16}" + "".join(f"{'n    acc   brier  ece':>26}" for _ in models))
    for question in SCOREBOARD_QUESTIONS:
        cells = []
        for model in models:
            summary = summarize(by_model[model], question)
            if summary is None:
                cells.append(f"{'-':>26}")
            else:
                cells.append(
                    f"{summary['n']:>7} {summary['accuracy']:6.3f} {summary['brier']:6.3f} {summary['ece']:5.3f}"
                )
        print(f"{question:<16}" + "".join(cells))


def _fetch(args: argparse.Namespace) -> None:
    import json

    from nba_laya.feed import fetch_json, playbyplay_url

    payload, _ = fetch_json(playbyplay_url(args.game_id))
    if payload is None:
        raise SystemExit(f"empty response for {args.game_id}")
    actions = payload.get("game", {}).get("actions", [])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"playbyplay_{args.game_id}.json"
    out.write_text(json.dumps(payload))
    print(f"saved {len(actions)} actions to {out}")


def _replay(args: argparse.Namespace) -> None:
    payload = load_payload(args.game_id, args.pbp)
    client = None
    if args.model == "laya":
        client = SystemOneClient(args.base_url, model="convaiinnovations/laya-multilingual")
    elif args.model == "jev":
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("TYPESAFE_API_KEY is not set")
        client = SystemOneClient("https://api.typesafe.ai", api_key=key, model="jev-latest")
    rows = replay(payload, model=args.model, client=client)
    text = format_scoreboard(rows)
    print(text)
    if args.out:
        write_log(rows, args.out)
        print(f"wrote {len(rows)} calls to {args.out}")
    elif not rows:
        print("no possession changes in this file")


if __name__ == "__main__":
    main()
