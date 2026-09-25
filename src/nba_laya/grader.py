"""Attach what actually happened. Windows stop at the period boundary."""

from __future__ import annotations

from nba_laya.actions import Action
from nba_laya.state import FIELD_GOALS, Decision


def grade(decisions: list[Decision], actions: list[Action]) -> None:
    final_home, final_away = _final_score(actions)
    boundaries = [decision.snapshot.action_index for decision in decisions]
    for index, decision in enumerate(decisions):
        snap = decision.snapshot
        labels: dict = {}
        if final_home != final_away:
            labels["winner"] = "home" if final_home > final_away else "away"
        possession_end = _window_end(actions, snap.action_index, boundaries, index, 1)
        labels["possession_scores"] = _team_scores(actions, snap.possession, snap.action_index, possession_end)
        score_type = _next_score_type(actions, snap.action_index)
        if score_type is not None:
            labels["score_type"] = score_type
        if snap.run:
            run_end = _window_end(actions, snap.action_index, boundaries, index, 3)
            labels["run_continues"] = _run_outscores(actions, snap, run_end)
        if snap.shooter_menu:
            shooter = _next_shooter(actions, snap.possession, snap.action_index)
            if shooter in snap.shooter_menu:
                labels["shooter"] = shooter
        if snap.comeback:
            labels["comeback"] = _retakes_lead(actions, snap)
        decision.labels = labels


def _window_end(actions: list[Action], start: int, boundaries: list[int], decision_index: int, count: int) -> int:
    later = [b for b in boundaries[decision_index + 1 :] if b > start]
    period = actions[start].period
    for action in actions[start + 1 :]:
        if action.period != period or action.is_period_end:
            period_end = action.index
            break
    else:
        period_end = len(actions) - 1
    if len(later) >= count:
        return min(later[count - 1], period_end)
    return period_end


def _points(action: Action) -> int:
    if not action.made_shot:
        return 0
    if action.action_type == "3pt":
        return 3
    if "free" in action.action_type:
        return 1
    return 2


def _team_scores(actions: list[Action], team: str, start: int, end: int) -> bool:
    return any(
        action.team_tricode == team and _points(action) > 0
        for action in actions[start + 1 : end + 1]
    )


def _next_score_type(actions: list[Action], start: int) -> str | None:
    period = actions[start].period
    for action in actions[start + 1 :]:
        if action.period != period or action.is_period_end:
            return None
        points = _points(action)
        if points == 3:
            return "three"
        if points == 2:
            return "two"
        if points == 1:
            return "free_throw"
    return None


def _next_shooter(actions: list[Action], team: str, start: int) -> str | None:
    period = actions[start].period
    for action in actions[start + 1 :]:
        if action.period != period or action.is_period_end:
            return None
        if action.action_type in FIELD_GOALS and action.team_tricode == team:
            return action.player_name or None
    return None


def _run_outscores(actions: list[Action], snapshot, end: int) -> bool:
    team = snapshot.run["team"]
    opp = snapshot.run["opponent"]
    ours = theirs = 0
    for action in actions[snapshot.action_index + 1 : end + 1]:
        points = _points(action)
        if action.team_tricode == team:
            ours += points
        elif action.team_tricode == opp:
            theirs += points
    return ours > theirs


def _retakes_lead(actions: list[Action], snapshot) -> bool:
    trailing_home = snapshot.comeback["team"] == snapshot.home
    for action in actions[snapshot.action_index + 1 :]:
        lead = action.score_home - action.score_away
        if (trailing_home and lead > 0) or (not trailing_home and lead < 0):
            return True
    return False


def _final_score(actions: list[Action]) -> tuple[int, int]:
    home = away = 0
    for action in actions:
        home = max(home, action.score_home)
        away = max(away, action.score_away)
    return home, away
