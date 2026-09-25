"""Walk the tape and build a snapshot at each possession change."""

from __future__ import annotations

from dataclasses import dataclass, field

from nba_laya.actions import Action
from nba_laya.clock import display_clock, elapsed_seconds, parse_clock, seconds_left_in_game
from nba_laya.wp import win_prob

TIMEOUTS_PER_GAME = 7
LAST_PLAYS = 8
RUN_MIN_POINTS = 6


@dataclass
class Snapshot:
    game_id: str
    action_number: int
    action_index: int
    game: str
    period: int
    clock: str
    seconds_left: float
    score: dict[str, int]
    home: str
    away: str
    possession: str | None
    run: dict | None
    last_3min: dict[str, int]
    timeouts_left: dict[str, int]
    last_timeout: dict | None
    fouls: dict[str, dict[str, int]]
    team_fouls_period: dict[str, int]
    on_floor: dict[str, list[str]]
    last_plays: list[str]
    win_prob_home: float

    def as_state(self) -> dict:
        state = {
            "game": self.game,
            "period": self.period,
            "clock": self.clock,
            "score": self.score,
            "possession": self.possession,
            "run": self.run,
            "last_3min": self.last_3min,
            "timeouts_left": self.timeouts_left,
            "last_timeout": self.last_timeout,
            "fouls": self.fouls,
            "team_fouls_period": self.team_fouls_period,
            "on_floor": self.on_floor,
            "last_plays": self.last_plays,
        }
        return {key: value for key, value in state.items() if value not in (None, {}, [])}


@dataclass
class Decision:
    snapshot: Snapshot
    labels: dict = field(default_factory=dict)


class GameWalk:
    def __init__(self, game_id: str, actions: list[Action]):
        self.game_id = game_id
        self.actions = actions
        self.home: str | None = None
        self.away: str | None = None
        self.team_of: dict[int, str] = {}
        self.possession: str | None = None
        self.score = {"home": 0, "away": 0}
        self.timeouts_used: dict[str, int] = {}
        self.last_timeout_elapsed: dict[str, float] = {}
        self.fouls: dict[str, dict[str, int]] = {}
        self.team_fouls: dict[str, int] = {}
        self.foul_period: int | None = None
        self.on_floor: dict[str, list[str]] = {}
        self.last_plays: list[str] = []
        self.scoring_events: list[tuple[float, str, int]] = []
        self.run_team: str | None = None
        self.run_points = 0
        self.run_team_possessions = 0
        self.possession_changes = 0
        self.decisions: list[Decision] = []

    def walk(self) -> list[Decision]:
        previous_possession: str | None = None
        for action in self.actions:
            self._apply(action)
            if self.home is None or self.away is None or self.possession is None:
                previous_possession = self.possession
                continue
            if previous_possession is None:
                previous_possession = self.possession
                continue
            if action.is_period_end:
                previous_possession = self.possession
                continue
            if self.possession != previous_possession:
                self.possession_changes += 1
                if self.possession == self.run_team:
                    self.run_team_possessions += 1
                self.decisions.append(Decision(snapshot=self.snapshot(action)))
            previous_possession = self.possession
        return self.decisions

    def snapshot(self, action: Action) -> Snapshot:
        assert self.home and self.away and self.possession
        elapsed = elapsed_seconds(action.period, action.clock, action.period_type)
        home_margin = self.score["home"] - self.score["away"]
        left = seconds_left_in_game(action.period, action.clock, action.period_type)
        names = {self.home: self.score["home"], self.away: self.score["away"]}
        run = None
        if self.run_team and self.run_points >= RUN_MIN_POINTS:
            opp = self.away if self.run_team == self.home else self.home
            run = {
                "team": self.run_team,
                "points": self.run_points,
                "opp_points": 0,
                "possessions": self.run_team_possessions,
                "opponent": opp,
            }
        last_timeout = None
        if self.last_timeout_elapsed:
            team, when = max(self.last_timeout_elapsed.items(), key=lambda item: item[1])
            last_timeout = {"team": team, "seconds_ago": int(elapsed - when)}
        recent = [
            (team, points)
            for at, team, points in self.scoring_events
            if elapsed - at <= 180
        ]
        last_3min = {
            self.home: sum(points for team, points in recent if team == self.home),
            self.away: sum(points for team, points in recent if team == self.away),
        }
        return Snapshot(
            game_id=self.game_id,
            action_number=action.action_number,
            action_index=action.index,
            game=f"{self.away} @ {self.home}",
            period=action.period,
            clock=display_clock(parse_clock(action.clock)),
            seconds_left=left,
            score=names,
            home=self.home,
            away=self.away,
            possession=self.possession,
            run=run,
            last_3min=last_3min,
            timeouts_left={
                team: max(0, TIMEOUTS_PER_GAME - self.timeouts_used.get(team, 0))
                for team in (self.home, self.away)
            },
            last_timeout=last_timeout,
            fouls={
                team: dict(players)
                for team, players in self.fouls.items()
                if players
            },
            team_fouls_period={
                team: self.team_fouls.get(team, 0) for team in (self.home, self.away)
            },
            on_floor={
                team: list(players)
                for team, players in self.on_floor.items()
                if players
            },
            last_plays=list(self.last_plays[-LAST_PLAYS:]),
            win_prob_home=win_prob(home_margin, left, self.possession == self.home),
        )

    def _apply(self, action: Action) -> None:
        if action.team_id and action.team_tricode:
            self.team_of[action.team_id] = action.team_tricode
        self._infer_sides(action)
        self._reset_period_fouls(action.period)
        if action.description:
            who = action.team_tricode or ""
            text = f"{who} {action.description}".strip()
            self.last_plays.append(text)

        points = self._points_on(action)
        scorer = self._side_of(action.team_tricode) if points else None
        if points and scorer:
            self.score[scorer] += points
            elapsed = elapsed_seconds(action.period, action.clock, action.period_type)
            self.scoring_events.append((elapsed, action.team_tricode, points))
            if self.run_team == action.team_tricode:
                self.run_points += points
            else:
                self.run_team = action.team_tricode
                self.run_points = points
                self.run_team_possessions = 1

        if action.is_timeout and action.team_tricode:
            self.timeouts_used[action.team_tricode] = self.timeouts_used.get(action.team_tricode, 0) + 1
            self.last_timeout_elapsed[action.team_tricode] = elapsed_seconds(
                action.period, action.clock, action.period_type
            )
        if action.counts_as_team_foul and action.team_tricode:
            self.team_fouls[action.team_tricode] = self.team_fouls.get(action.team_tricode, 0) + 1
            if action.player_name and action.foul_personal_total:
                self.fouls.setdefault(action.team_tricode, {})[action.player_name] = action.foul_personal_total
        if action.is_substitution_out and action.team_tricode and action.player_name:
            floor = self.on_floor.setdefault(action.team_tricode, [])
            if action.player_name in floor:
                floor.remove(action.player_name)
        elif action.action_type == "substitution" and action.sub_type == "in" and action.player_name:
            floor = self.on_floor.setdefault(action.team_tricode, [])
            if action.player_name not in floor:
                floor.append(action.player_name)

        self.possession = self._possession_after(action)

    def _points_on(self, action: Action) -> int:
        if not action.made_shot:
            return 0
        if action.action_type == "3pt":
            return 3
        if action.action_type in {"freethrow", "free throw", "made freethrow"}:
            return 1
        if action.action_type in {"2pt", "made shot"}:
            return 2
        if action.points_total in {1, 2, 3}:
            return action.points_total
        return 0

    def _possession_after(self, action: Action) -> str | None:
        if action.ends_possession_on_make and action.team_tricode and self.home and self.away:
            return self.away if action.team_tricode == self.home else self.home
        if action.possession and action.possession in self.team_of:
            return self.team_of[action.possession]
        return self.possession

    def _infer_sides(self, action: Action) -> None:
        previous_home = self._last_score_home
        previous_away = self._last_score_away
        if action.score_home > previous_home and action.team_tricode:
            self.home = action.team_tricode
        if action.score_away > previous_away and action.team_tricode:
            self.away = action.team_tricode
        self._last_score_home = action.score_home
        self._last_score_away = action.score_away
        if self.home and not self.away and action.team_tricode and action.team_tricode != self.home:
            self.away = action.team_tricode
        if self.away and not self.home and action.team_tricode and action.team_tricode != self.away:
            self.home = action.team_tricode

    _last_score_home = 0
    _last_score_away = 0

    def _side_of(self, tricode: str) -> str | None:
        if tricode == self.home:
            return "home"
        if tricode == self.away:
            return "away"
        return None

    def _reset_period_fouls(self, period: int) -> None:
        if self.foul_period != period:
            self.team_fouls = {}
            self.foul_period = period


def load_game(payload: dict) -> tuple[str, list[Action]]:
    from nba_laya.actions import parse_actions

    game = payload.get("game", payload)
    game_id = str(game.get("gameId") or payload.get("gameId") or "unknown")
    raw = game.get("actions") or payload.get("actions") or []
    return game_id, parse_actions(raw)
