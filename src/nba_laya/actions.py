"""Normalize one live play-by-play action.

The live CDN object uses actionType values like ``2pt`` and ``timeout``.
Older feeds use ``Made Shot``. Both are accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FT_PROGRESS = re.compile(r"(\d+)\s*of\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Action:
    index: int
    action_number: int
    clock: str
    period: int
    period_type: str
    action_type: str
    sub_type: str
    team_id: int | None
    team_tricode: str
    person_id: int | None
    player_name: str
    description: str
    possession: int | None
    score_home: int
    score_away: int
    shot_result: str
    points_total: int
    foul_personal_total: int
    time_actual: str | None

    @property
    def is_period_end(self) -> bool:
        return self.action_type == "period" and self.sub_type in {"end", "ended"}

    @property
    def is_timeout(self) -> bool:
        return self.action_type == "timeout"

    @property
    def is_substitution_out(self) -> bool:
        if self.action_type != "substitution":
            return False
        if self.sub_type in {"out", "substitutionout"}:
            return True
        return " for " in self.description.lower() and self.sub_type != "in"

    @property
    def is_foul(self) -> bool:
        return self.action_type == "foul"

    @property
    def counts_as_team_foul(self) -> bool:
        if not self.is_foul:
            return False
        ignored = {"technical", "defensive3second", "flagrant"}
        # Flagrant personal still counts. A plain technical does not.
        if self.sub_type in ignored and "personal" not in self.sub_type:
            return False
        return True

    @property
    def made_shot(self) -> bool:
        if self.shot_result == "made":
            return True
        if self.action_type in {"made shot", "made freethrow"}:
            return True
        return False

    @property
    def ends_possession_on_make(self) -> bool:
        """Made field goals flip the ball. A free throw flips only when it is the last."""
        if not self.made_shot:
            return False
        if self.action_type in {"2pt", "3pt", "made shot"}:
            return True
        if self.action_type in {"freethrow", "free throw", "made freethrow"}:
            match = _FT_PROGRESS.search(self.sub_type) or _FT_PROGRESS.search(self.description)
            if match and int(match.group(1)) < int(match.group(2)):
                return False
            return True
        return False


def _int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value) -> int | None:
    if value is None or value == "" or value == 0:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_action(raw: dict, index: int) -> Action:
    action_type = str(raw.get("actionType") or "").strip().lower()
    sub_type = str(raw.get("subType") or "").strip().lower()
    return Action(
        index=index,
        action_number=_int(raw.get("actionNumber"), index),
        clock=str(raw.get("clock") or ""),
        period=_int(raw.get("period"), 1),
        period_type=str(raw.get("periodType") or "REGULAR"),
        action_type=action_type,
        sub_type=sub_type,
        team_id=_optional_int(raw.get("teamId")),
        team_tricode=str(raw.get("teamTricode") or "").upper(),
        person_id=_optional_int(raw.get("personId")),
        player_name=str(raw.get("playerNameI") or raw.get("playerName") or ""),
        description=str(raw.get("description") or "").strip(),
        possession=_optional_int(raw.get("possession")),
        score_home=_int(raw.get("scoreHome")),
        score_away=_int(raw.get("scoreAway")),
        shot_result=str(raw.get("shotResult") or "").strip().lower(),
        points_total=_int(raw.get("pointsTotal")),
        foul_personal_total=_int(raw.get("foulPersonalTotal")),
        time_actual=raw.get("timeActual"),
    )


def parse_actions(raw_actions: list[dict]) -> list[Action]:
    return [parse_action(raw, i) for i, raw in enumerate(raw_actions)]
