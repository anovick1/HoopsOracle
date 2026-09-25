"""Pull whole seasons of finished games from the CDN.

Game ids are ``0`` + type + two-digit season + five-digit number.
Type 001 preseason, 002 regular season, 004 playoffs (round, series, game).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from nba_laya.feed import FeedError, fetch_json, playbyplay_url

REGULAR_SEASON_GAMES = 1230
PLAYOFF_SERIES_PER_ROUND = {1: 8, 2: 4, 3: 2, 4: 1}


def regular_season_ids(season: int) -> list[str]:
    return [f"002{season:02d}{n:05d}" for n in range(1, REGULAR_SEASON_GAMES + 1)]


def playoff_ids(season: int) -> list[str]:
    ids = []
    for rnd, series_count in PLAYOFF_SERIES_PER_ROUND.items():
        for series in range(1, series_count + 1):
            for game in range(1, 8):
                ids.append(f"004{season:02d}00{rnd}{series - 1}{game}")
    return ids


def fetch_many(game_ids: list[str], out_dir: Path, pause: float = 0.6, log=print) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"saved": 0, "skipped": 0, "missing": 0, "failed": 0}
    for gid in game_ids:
        target = out_dir / f"playbyplay_{gid}.json"
        if target.exists():
            counts["skipped"] += 1
            continue
        try:
            payload, _ = fetch_json(playbyplay_url(gid))
        except FeedError as exc:
            text = str(exc)
            if text.startswith("404"):
                counts["missing"] += 1
            elif text.startswith("403"):
                log(f"{gid}: CDN denied the request. Stopping so the block does not harden.")
                counts["failed"] += 1
                break
            else:
                counts["failed"] += 1
                log(f"{gid} failed: {text[:80]}")
                time.sleep(pause * 5)
            time.sleep(pause)
            continue
        actions = (payload or {}).get("game", {}).get("actions", [])
        if len(actions) < 50:
            counts["missing"] += 1
            time.sleep(pause)
            continue
        target.write_text(json.dumps(payload))
        counts["saved"] += 1
        if counts["saved"] % 50 == 0:
            log(f"saved {counts['saved']} (skipped {counts['skipped']}, missing {counts['missing']})")
        time.sleep(pause)
    return counts
