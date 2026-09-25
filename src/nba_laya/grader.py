"""Attach what actually happened. Windows stop at the period boundary."""

from __future__ import annotations

from nba_laya.actions import Action
from nba_laya.state import Decision
from nba_laya.wp import swing_bucket


def grade(decisions: list[Decision], actions: list[Action]) -> None:
    final_home, final_away = _final_score(actions)
    boundaries = [decision.snapshot.action_index for decision in decisions]
    previous_wp = None
    for index, decision in enumerate(decisions):
        snap = decision.snapshot
        labels: dict = {}
        end = _window_end(actions, snap.action_index, boundaries, index, 2)
        labels["timeout_next2"] = _timeout_in(actions, snap.action_index, end)
        labels["sub_next2"] = _sub_out(actions, snap, end)
        next_score = _next_score(actions, snap)
        if next_score is not None:
            labels["next_score"] = next_score
        if snap.run:
            run_end = _window_end(actions, snap.action_index, boundaries, index, 3)
            labels["run_continues"] = _run_outscores(actions, snap, run_end)
        if final_home != final_away:
            labels["winner"] = "home" if final_home > final_away else "away"
        bucket = swing_bucket(previous_wp, snap.win_prob_home)
        if bucket is not None:
            labels["swing"] = bucket
        previous_wp = snap.win_prob_home
        decision.labels = labels


def _window_end(actions: list[Action], start: int, boundaries: list[int], decision_index: int, count: int) -> int:
    later = [b for b in boundaries[decision_index + 1 :] if b > start]
    if len(later) >= count:
        return later[count - 1]
    period = actions[start].period
    for action in actions[start + 1 :]:
        if action.period != period or action.is_period_end:
            return action.index
    return len(actions) - 1


def _timeout_in(actions: list[Action], start: int, end: int) -> bool:
    return any(action.is_timeout for action in actions[start + 1 : end + 1])


def _players_in_foul_trouble(snapshot) -> set[str]:
    names = set()
    for players in snapshot.fouls.values():
        for name, fouls in players.items():
            if fouls >= 4:
                names.add(name)
    return names


def _sub_out(actions: list[Action], snapshot, end: int) -> bool:
    trouble = _players_in_foul_trouble(snapshot)
    if not trouble:
        return False
    for action in actions[snapshot.action_index + 1 : end + 1]:
        if action.is_substitution_out and action.player_name in trouble:
            return True
    return False


def _next_score(actions: list[Action], snapshot) -> str | None:
    period = actions[snapshot.action_index].period
    for action in actions[snapshot.action_index + 1 :]:
        if action.period != period or action.is_period_end:
            return None
        if action.made_shot and action.team_tricode:
            if action.team_tricode == snapshot.home:
                return "home"
            if action.team_tricode == snapshot.away:
                return "away"
    return None


def _run_outscores(actions: list[Action], snapshot, end: int) -> bool:
    team = snapshot.run["team"]
    opp = snapshot.run["opponent"]
    ours = theirs = 0
    for action in actions[snapshot.action_index + 1 : end + 1]:
        if not action.made_shot:
            continue
        points = 3 if action.action_type == "3pt" else 1 if "free" in action.action_type else 2
        if action.team_tricode == team:
            ours += points
        elif action.team_tricode == opp:
            theirs += points
    return ours > theirs


def _final_score(actions: list[Action]) -> tuple[int, int]:
    home = away = 0
    for action in actions:
        home = max(home, action.score_home)
        away = max(away, action.score_away)
    return home, away
