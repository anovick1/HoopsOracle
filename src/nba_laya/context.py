"""Pregame context: what was known about each team and player before tip-off.

Built from the saved tapes in date order. Every number for a game uses only
games that finished before it, so training never sees the future.

Early in a season the current numbers are a handful of games, so they are
shrunk toward last season's: value = (n * current + K * prior) / (n + K).

Team: offensive / defensive / net rating (per 100 possessions), pace, three-point
and free-throw rates for and against, record, rest.
Player: share of team shots, share in clutch (last 5 min, within 8), three rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from nba_laya.actions import Action, parse_actions
from nba_laya.clock import seconds_left_in_game

RATING_PRIOR_GAMES = 10
SHARE_PRIOR_GAMES = 5
# Clutch shots are rare; a player's clutch share leans on their overall share
# until this many team clutch attempts have been seen with them shooting.
CLUTCH_PRIOR_ATTEMPTS = 20
CLUTCH_SECONDS = 300
CLUTCH_MARGIN = 8
# Tape timestamps are UTC. Shifting by six hours puts late tips on their US date.
LOCAL_OFFSET = timedelta(hours=6)
FIELD_GOALS = {"2pt", "3pt"}


@dataclass
class TeamGame:
    points: int = 0
    fga: int = 0
    fga3: int = 0
    fta: int = 0
    oreb: int = 0
    tov: int = 0
    clutch_fga: int = 0
    # name -> [fga, fga3, clutch_fga]
    players: dict[str, list[int]] = field(default_factory=dict)

    @property
    def possessions(self) -> float:
        return max(1.0, self.fga + 0.44 * self.fta - self.oreb + self.tov)


@dataclass
class GameSummary:
    game_id: str
    season: int
    date: str
    teams: tuple[str, str]
    box: dict[str, TeamGame]

    @property
    def order_key(self) -> tuple[str, str]:
        return (self.date, self.game_id)

    @property
    def points(self) -> dict[str, int]:
        return {t: b.points for t, b in self.box.items()}


def _is_clutch(a: Action) -> bool:
    if a.period < 4:
        return False
    left = seconds_left_in_game(a.period, a.clock, a.period_type)
    return left <= CLUTCH_SECONDS and abs(a.score_home - a.score_away) <= CLUTCH_MARGIN


def summarize_game(payload: dict) -> GameSummary | None:
    game = payload.get("game", payload)
    game_id = str(game.get("gameId"))
    actions = parse_actions(game.get("actions") or [])
    first = next((a.time_actual for a in actions if a.time_actual), None)
    teams = sorted({a.team_tricode for a in actions if a.team_tricode})
    if not first or len(teams) != 2:
        return None
    when = datetime.fromisoformat(first.replace("Z", "+00:00")) - LOCAL_OFFSET
    box = {t: TeamGame() for t in teams}
    for a in actions:
        b = box.get(a.team_tricode)
        if b is None:
            continue
        if a.made_shot:
            b.points += 3 if a.action_type == "3pt" else 1 if "free" in a.action_type else 2
        if a.action_type in FIELD_GOALS:
            b.fga += 1
            three = a.action_type == "3pt"
            clutch = _is_clutch(a)
            b.fga3 += int(three)
            b.clutch_fga += int(clutch)
            if a.player_name:
                rec = b.players.setdefault(a.player_name, [0, 0, 0])
                rec[0] += 1
                rec[1] += int(three)
                rec[2] += int(clutch)
        elif a.action_type == "freethrow":
            b.fta += 1
        elif a.action_type == "rebound" and a.sub_type == "offensive":
            b.oreb += 1
        elif a.action_type == "turnover":
            b.tov += 1
    return GameSummary(
        game_id=game_id,
        season=int(game_id[3:5]),
        date=when.date().isoformat(),
        teams=(teams[0], teams[1]),
        box=box,
    )


@dataclass
class TeamLine:
    games: int = 0
    wins: int = 0  # regular season only, so the record reads like the standings
    losses: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    poss_for: float = 0.0
    poss_against: float = 0.0
    fga: int = 0
    fga3: int = 0
    fta: int = 0
    opp_fga: int = 0
    opp_fga3: int = 0
    opp_fta: int = 0
    last_date: str | None = None

    def stats(self) -> dict[str, float]:
        if not self.games:
            return {}
        return {
            "off_rating": 100.0 * self.points_for / max(1.0, self.poss_for),
            "def_rating": 100.0 * self.points_against / max(1.0, self.poss_against),
            "pace": self.poss_for / self.games,
            "three_rate": self.fga3 / max(1, self.fga),
            "ft_rate": self.fta / max(1, self.fga),
            "opp_three_rate": self.opp_fga3 / max(1, self.opp_fga),
            "opp_ft_rate": self.opp_fta / max(1, self.opp_fga),
        }


# League-average fallbacks for a team with no history at all.
LEAGUE = {"off_rating": 113.0, "def_rating": 113.0, "pace": 99.0, "three_rate": 0.40,
          "ft_rate": 0.22, "opp_three_rate": 0.40, "opp_ft_rate": 0.22}


@dataclass
class PlayerLine:
    fga: int = 0
    fga3: int = 0
    team_fga: int = 0  # team attempts in games the player shot in
    clutch_fga: int = 0
    team_clutch_fga: int = 0
    games: int = 0


class Ledger:
    """Running state of the league. Feed games in date order with `add`."""

    def __init__(self) -> None:
        self.season: int | None = None
        self.teams: dict[str, TeamLine] = {}
        self.players: dict[str, PlayerLine] = {}
        self.prior_team: dict[str, dict[str, float]] = {}
        self.prior_player: dict[str, dict[str, float]] = {}

    def _roll(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self.prior_team = {t: line.stats() for t, line in self.teams.items() if line.games}
            self.prior_player = {name: p for name, p in ((n, self._player_raw(n)) for n in self.players) if p}
            self.teams = {}
            self.players = {}
        self.season = season

    def _shrink(self, current: dict, prior: dict, n: float, k: float, keys) -> dict:
        out = {}
        for key in keys:
            cur, pri = current.get(key), prior.get(key, LEAGUE.get(key))
            if cur is None and pri is None:
                continue
            if cur is None:
                out[key] = pri
            elif pri is None:
                out[key] = cur
            else:
                out[key] = (n * cur + k * pri) / (n + k)
        return out

    def team_context(self, team: str, date: str) -> dict:
        line = self.teams.get(team, TeamLine())
        stats = self._shrink(line.stats(), self.prior_team.get(team, {}), line.games, RATING_PRIOR_GAMES, LEAGUE)
        # Whole numbers and no record on purpose. Pregame context is constant
        # within a game, and so is the winner label; precise values make each
        # game identifiable and a model memorizes results instead of basketball.
        context = {
            "net": round(stats["off_rating"] - stats["def_rating"]),
            "off": round(stats["off_rating"]),
            "def": round(stats["def_rating"]),
            "pace": round(stats["pace"]),
            "three": round(stats["three_rate"], 2),
            "ft": round(stats["ft_rate"], 2),
            "opp_three": round(stats["opp_three_rate"], 2),
        }
        if line.last_date:
            context["rest"] = (datetime.fromisoformat(date) - datetime.fromisoformat(line.last_date)).days
        return context

    def _player_raw(self, name: str) -> dict[str, float]:
        p = self.players.get(name)
        if not p or not p.team_fga:
            return {}
        raw = {"share": p.fga / p.team_fga, "three_rate": p.fga3 / max(1, p.fga)}
        if p.team_clutch_fga:
            raw["clutch_share"] = p.clutch_fga / p.team_clutch_fga
            raw["_clutch_n"] = p.team_clutch_fga
        return raw

    def player(self, name: str) -> dict | None:
        """Pregame profile for one player, or None with no history at all."""
        p = self.players.get(name)
        raw = self._player_raw(name)
        prior = self.prior_player.get(name, {})
        if not raw and not prior:
            return None
        n = p.games if p else 0
        out = self._shrink(raw, prior, n, SHARE_PRIOR_GAMES, ("share", "three_rate"))
        share = out.get("share")
        # Clutch share: shrink toward the overall share until enough clutch attempts.
        clutch_n = raw.get("_clutch_n", 0) + (CLUTCH_PRIOR_ATTEMPTS if "clutch_share" in prior else 0)
        clutch_sum = raw.get("clutch_share", 0) * raw.get("_clutch_n", 0)
        if "clutch_share" in prior:
            clutch_sum += prior["clutch_share"] * CLUTCH_PRIOR_ATTEMPTS
        if share is not None:
            out["clutch_share"] = (clutch_sum + share * CLUTCH_PRIOR_ATTEMPTS) / (clutch_n + CLUTCH_PRIOR_ATTEMPTS)
        return {k: round(v, 3) for k, v in out.items() if not k.startswith("_")}

    def pregame(self, summary_or_teams, date: str, season: int) -> dict:
        """Context for a game about to start. Call before `add` for that game."""
        self._roll(season)
        teams = summary_or_teams.teams if isinstance(summary_or_teams, GameSummary) else summary_or_teams
        return {"teams": {t: self.team_context(t, date) for t in teams}}

    def add(self, g: GameSummary) -> None:
        self._roll(g.season)
        a, b = g.teams
        for team, opp in ((a, b), (b, a)):
            mine, theirs = g.box[team], g.box[opp]
            line = self.teams.setdefault(team, TeamLine())
            line.games += 1
            if g.game_id.startswith("002"):
                won = mine.points > theirs.points
                line.wins += int(won)
                line.losses += int(not won)
            line.points_for += mine.points
            line.points_against += theirs.points
            line.poss_for += mine.possessions
            line.poss_against += theirs.possessions
            line.fga += mine.fga
            line.fga3 += mine.fga3
            line.fta += mine.fta
            line.opp_fga += theirs.fga
            line.opp_fga3 += theirs.fga3
            line.opp_fta += theirs.fta
            line.last_date = g.date
            for name, (fga, fga3, clutch) in mine.players.items():
                p = self.players.setdefault(name, PlayerLine())
                p.fga += fga
                p.fga3 += fga3
                p.team_fga += mine.fga
                p.clutch_fga += clutch
                p.team_clutch_fga += mine.clutch_fga
                p.games += 1


def load_summaries(dirs: list[Path], log=print) -> dict[str, GameSummary]:
    summaries = {}
    for d in dirs:
        for path in sorted(d.glob("playbyplay_*.json")):
            s = summarize_game(json.loads(path.read_text()))
            if s:
                summaries[s.game_id] = s
    log(f"summarized {len(summaries)} games from {len(dirs)} folders")
    return summaries


@dataclass
class Pregame:
    """What the engine is allowed to know at tip-off for one game."""

    teams: dict[str, dict]
    date: str
    player: object  # callable name -> profile dict | None, frozen at tip-off


def in_date_order(summaries: dict[str, GameSummary]):
    """Yield (summary, Pregame) oldest first. The game is added to the ledger
    only after the consumer resumes, so everything computed from the Pregame
    during that step sees games strictly before this one."""
    ledger = Ledger()
    for g in sorted(summaries.values(), key=lambda s: s.order_key):
        teams = ledger.pregame(g, g.date, g.season)["teams"]
        yield g, Pregame(teams=teams, date=g.date, player=ledger.player)
        ledger.add(g)


def ledger_through(summaries: dict[str, GameSummary]) -> Ledger:
    """Ledger after every saved game. Live games read their pregame from this."""
    ledger = Ledger()
    for g in sorted(summaries.values(), key=lambda s: s.order_key):
        ledger.add(g)
    return ledger
