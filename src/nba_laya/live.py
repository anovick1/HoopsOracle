"""Poll live games and decide at each possession change.

One poller per live game. Each poll downloads the full tape (the CDN ignores
If-None-Match, so a poll is ~450 KB). New actions are the ones past the length
we have already walked, in the NBA's own array order. The GameWalk is fed
incrementally, so a snapshot uses only actions already on the tape.

Caveat: if the NBA edits or inserts an action earlier in the tape mid-game,
positional tracking misses it. Post-game grading re-walks the full file, so
the scoreboard is never affected; only the live call for that possession is.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nba_laya.actions import parse_action
from nba_laya.client import SystemOneClient, SystemOneError
from nba_laya.feed import FeedError, fetch_json, playbyplay_url, scoreboard_url
from nba_laya.questions import build_questions
from nba_laya.rules import answer
from nba_laya.state import Decision, GameWalk

LIVE_STATUS = 2  # gameStatus: 1 scheduled, 2 in progress, 3 final


def live_games() -> list[dict]:
    payload, _ = fetch_json(scoreboard_url())
    board = (payload or {}).get("scoreboard", {})
    return [
        {"game_id": g["gameId"], "date": board.get("gameDate"),
         "home": g["homeTeam"]["teamTricode"], "away": g["awayTeam"]["teamTricode"]}
        for g in board.get("games", [])
        if g.get("gameStatus") == LIVE_STATUS
    ]


def live_game_ids() -> list[tuple[str, str]]:
    return [(g["game_id"], f"{g['away']} @ {g['home']}") for g in live_games()]


def load_ledger(data_dir: Path = Path("data"), log=print):
    """Pregame ratings through every saved tape. Refresh nightly with make season."""
    from nba_laya.context import ledger_through, load_summaries

    dirs = sorted(p for p in data_dir.glob("season_*") if p.is_dir())
    if not dirs:
        log("no saved seasons; live calls run without team strength or shot share")
        return None
    return ledger_through(load_summaries(dirs, log=log))


def pregame_for(ledger, game: dict):
    from nba_laya.context import Pregame

    if ledger is None or not game.get("date"):
        return None
    season = int(game["game_id"][3:5])
    teams = ledger.pregame((game["away"], game["home"]), game["date"], season)["teams"]
    return Pregame(teams=teams, date=game["date"], player=ledger.player)


@dataclass
class GamePoller:
    game_id: str
    model: str = "rules"
    client: SystemOneClient | None = None
    log_path: Path | None = None
    pregame: object = None
    etag: str | None = None
    last_action_number: int = -1
    walk: GameWalk = field(init=False)
    previous_possession: str | None = None
    previous_wp: float | None = None
    polls: int = 0
    unchanged: int = 0

    def __post_init__(self) -> None:
        self.walk = GameWalk(self.game_id, [], pregame=self.pregame)

    def poll(self) -> list[dict]:
        """Fetch once. Return the decision rows produced by new actions."""
        self.polls += 1
        seen_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        payload, self.etag = fetch_json(playbyplay_url(self.game_id), etag=self.etag)
        if payload is None:
            self.unchanged += 1
            return []
        raw_actions = payload.get("game", {}).get("actions", [])
        # The tape is in the NBA's display order, which is not actionNumber order:
        # substitutions and timeouts are inserted with out-of-sequence numbers.
        # Track by position so live and replay walk the same sequence.
        new = raw_actions[len(self.walk.actions):]
        rows = []
        for raw in new:
            action = parse_action(raw, index=len(self.walk.actions))
            self.walk.actions.append(action)
            self.last_action_number = max(self.last_action_number, action.action_number)
            row = self._step(action, seen_at)
            if row:
                rows.append(row)
        return rows

    def _step(self, action, seen_at: str) -> dict | None:
        walk = self.walk
        walk._apply(action)
        if walk.home is None or walk.away is None or walk.possession is None:
            self.previous_possession = walk.possession
            return None
        if self.previous_possession is None or action.is_period_end:
            self.previous_possession = walk.possession
            return None
        if walk.possession == self.previous_possession:
            return None
        self.previous_possession = walk.possession
        walk.possession_changes += 1
        if walk.possession == walk.run_team:
            walk.run_team_possessions += 1
        snap = walk.snapshot(action)
        walk.decisions.append(Decision(snapshot=snap))
        questions = build_questions(snap)
        t0 = time.time()
        try:
            if self.model == "rules" or self.client is None:
                answers = answer(snap, self.previous_wp, questions)
            else:
                answers = self.client.system_one(snap.as_state(), questions)
        except SystemOneError as exc:
            answers = {"_error": str(exc)}
        latency_ms = round((time.time() - t0) * 1000)
        self.previous_wp = snap.win_prob_home
        row = {
            "game_id": self.game_id,
            "action_number": snap.action_number,
            "time_actual": action.time_actual,
            "seen_at": seen_at,
            "model": self.model,
            "latency_ms": latency_ms,
            "snapshot": snap.as_state(),
            "win_prob_home": round(snap.win_prob_home, 4),
            "questions": questions,
            "answers": answers,
        }
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        return row


def run_live(interval: float = 4.0, model: str = "rules", base_url: str = "http://127.0.0.1:8000",
             log_dir: Path = Path("logs/live"), once: bool = False, log=print) -> None:
    client = SystemOneClient(base_url, model="convaiinnovations/laya-multilingual") if model == "laya" else None
    ledger = load_ledger(log=log)
    pollers: dict[str, GamePoller] = {}
    while True:
        try:
            games = live_games()
        except FeedError as exc:
            log(f"scoreboard: {exc}")
            games = []
        live = [(g["game_id"], f"{g['away']} @ {g['home']}") for g in games]
        for game in games:
            gid = game["game_id"]
            if gid not in pollers:
                log(f"tracking {game['away']} @ {game['home']} ({gid})")
                pollers[gid] = GamePoller(gid, model=model, client=client, log_path=log_dir / f"{model}_{gid}.jsonl",
                                          pregame=pregame_for(ledger, game))
        for gid, poller in list(pollers.items()):
            try:
                rows = poller.poll()
            except FeedError as exc:
                log(f"{gid}: {exc}")
                continue
            for row in rows:
                s = row["snapshot"]
                log(f"{s['game']} P{s['period']} {s['clock']} {s['score']} -> {_brief(row['answers'])} ({row['latency_ms']} ms)")
            if gid not in dict(live) and poller.polls > 3:
                log(f"{gid} finished after {poller.polls} polls ({poller.unchanged} unchanged)")
                del pollers[gid]
        if once:
            return
        time.sleep(interval)


def _brief(answers: dict) -> str:
    parts = []
    for qid, a in answers.items():
        if qid.startswith("_"):
            return a
        if a.get("type") == "noul":
            parts.append(f"{qid}={a['noul']:.2f}")
        elif a.get("type") == "choice":
            parts.append(f"{qid}={a['choice']}:{a['probabilities'][a['choice']]:.2f}")
    return " ".join(parts)
