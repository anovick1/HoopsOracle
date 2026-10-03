"""Per-player lines from the tape, at any point in the game.

Points, shots, free throws, fouls are events. Rebounds and assists ride along
in the description text as cumulative counters ("REBOUND (Off:1 Def:4)",
"(K. Towns 3 AST)"). Minutes come from substitutions, seeded with the
boxscore's starters; between-period changes are logged at 12:00 of the new period.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nba_laya.actions import Action
from nba_laya.clock import parse_clock, period_length

_REB = re.compile(r"\(Off:(\d+) Def:(\d+)\)")
_AST = re.compile(r"\(([A-Z]\. [^()]+?) (\d+) AST\)")


@dataclass
class Line:
    team: str
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    tpm: int = 0
    tpa: int = 0
    ftm: int = 0
    fta: int = 0
    reb: int = 0
    ast: int = 0
    fouls: int = 0
    seconds: float = 0.0
    on_court: bool = False
    stint_start: float | None = None  # elapsed seconds when the current stint began

    @property
    def minutes(self) -> int:
        return int(round(self.seconds / 60))

    def as_dict(self) -> dict:
        return {
            "pts": self.pts, "fg": f"{self.fgm}/{self.fga}", "3p": f"{self.tpm}/{self.tpa}",
            "ft": f"{self.ftm}/{self.fta}", "reb": self.reb, "ast": self.ast,
            "pf": self.fouls, "min": self.minutes,
        }


class StatTracker:
    def __init__(self, starters: dict[str, list[str]] | None = None):
        self.lines: dict[str, Line] = {}
        self.elapsed = 0.0
        self.period = 0
        if starters:
            for team, names in starters.items():
                for name in names:
                    line = self._line(name, team)
                    line.on_court = True
                    line.stint_start = 0.0

    def _line(self, name: str, team: str) -> Line:
        line = self.lines.get(name)
        if line is None:
            line = self.lines[name] = Line(team=team)
        return line

    def _elapsed(self, action: Action) -> float:
        prior = sum(period_length(p, "REGULAR" if p <= 4 else "OVERTIME") for p in range(1, action.period))
        length = period_length(action.period, action.period_type)
        return prior + (length - min(parse_clock(action.clock), length))

    def apply(self, action: Action) -> None:
        self.elapsed = self._elapsed(action)
        self.period = action.period
        name, team = action.player_name, action.team_tricode
        if action.action_type == "substitution" and name and team:
            line = self._line(name, team)
            if action.sub_type == "in":
                if not line.on_court:
                    line.on_court = True
                    line.stint_start = self.elapsed
            elif action.is_substitution_out:
                self._close_stint(line)
            return
        if not name or not team:
            return
        line = self._line(name, team)
        # Anyone recording an on-court action without a logged sub-in is on the floor.
        if not line.on_court and action.action_type in ("2pt", "3pt", "freethrow", "rebound", "turnover", "steal", "block", "foul"):
            line.on_court = True
            line.stint_start = self.elapsed
        if action.action_type in ("2pt", "3pt"):
            line.fga += 1
            if action.action_type == "3pt":
                line.tpa += 1
            if action.made_shot:
                line.fgm += 1
                line.pts += 3 if action.action_type == "3pt" else 2
                if action.action_type == "3pt":
                    line.tpm += 1
        elif action.action_type == "freethrow":
            line.fta += 1
            if action.made_shot:
                line.ftm += 1
                line.pts += 1
        elif action.action_type == "rebound":
            m = _REB.search(action.description)
            if m:
                line.reb = int(m.group(1)) + int(m.group(2))
        elif action.action_type == "foul" and action.foul_personal_total:
            line.fouls = action.foul_personal_total
        if action.made_shot:
            m = _AST.search(action.description)
            if m:
                passer = self._line(m.group(1).strip(), team)
                passer.ast = max(passer.ast, int(m.group(2)))

    def _close_stint(self, line: Line) -> None:
        if line.on_court and line.stint_start is not None:
            line.seconds += max(0.0, self.elapsed - line.stint_start)
        line.on_court = False
        line.stint_start = None

    def snapshot_lines(self) -> dict[str, dict]:
        """Current lines, with open stints counted up to now."""
        out = {}
        for name, line in self.lines.items():
            seconds = line.seconds
            if line.on_court and line.stint_start is not None:
                seconds += max(0.0, self.elapsed - line.stint_start)
            d = line.as_dict()
            d["min"] = int(round(seconds / 60))
            d["team"] = line.team
            d["on_court"] = line.on_court
            out[name] = d
        return out


def boxscore_starters(box: dict) -> dict[str, list[str]]:
    game = box.get("game", box)
    out = {}
    for side in ("homeTeam", "awayTeam"):
        team = game[side]
        out[team["teamTricode"]] = [p["nameI"] for p in team["players"] if p.get("starter") in (1, "1", True)]
    return out


def boxscore_lines(box: dict) -> dict[str, dict]:
    """Final lines from the boxscore: name -> stats, with team and starter flag."""
    game = box.get("game", box)
    out = {}
    for side in ("homeTeam", "awayTeam"):
        team = game[side]
        for p in team["players"]:
            s = p.get("statistics") or {}
            out[p["nameI"]] = {
                "team": team["teamTricode"], "starter": p.get("starter") in (1, "1", True),
                "played": p.get("played") in (1, "1", True),
                "pts": s.get("points", 0), "reb": s.get("reboundsTotal", 0), "ast": s.get("assists", 0),
                "fga": s.get("fieldGoalsAttempted", 0), "tpa": s.get("threePointersAttempted", 0),
                "fta": s.get("freeThrowsAttempted", 0), "min": _minutes(s.get("minutes")),
            }
    return out


def _minutes(iso: str | None) -> float:
    if not iso:
        return 0.0
    m = re.match(r"PT(\d+)M([\d.]+)S", iso)
    return round(int(m.group(1)) + float(m.group(2)) / 60, 1) if m else 0.0
