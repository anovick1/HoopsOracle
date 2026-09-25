"""NBA live CDN.

The CDN sits behind Akamai, which rejects the TLS handshake of plain Python
and macOS curl with an "Access Denied" page. ``curl_cffi`` impersonates a
Chrome handshake, and the nba.com Referer is still required.
"""

from __future__ import annotations

import json
from pathlib import Path

from curl_cffi import requests

CDN = "https://cdn.nba.com/static/json/liveData"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}
IMPERSONATE = "chrome"


class FeedError(RuntimeError):
    pass


def fetch_json(url: str, etag: str | None = None, timeout: float = 20) -> tuple[dict | None, str | None]:
    """Return (body, etag). Body is None when the CDN says nothing changed."""
    headers = dict(HEADERS)
    if etag:
        headers["If-None-Match"] = etag
    try:
        response = requests.get(url, headers=headers, impersonate=IMPERSONATE, timeout=timeout)
    except requests.RequestsError as exc:
        raise FeedError(f"could not reach {url}: {exc}") from exc
    if response.status_code == 304:
        return None, etag
    if response.status_code != 200:
        body = response.text[:180]
        raise FeedError(f"{response.status_code} from {url}: {body}")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise FeedError(f"non-JSON body from {url}: {response.text[:120]!r}") from exc
    return payload, response.headers.get("etag")


def playbyplay_url(game_id: str) -> str:
    return f"{CDN}/playbyplay/playbyplay_{game_id}.json"


def scoreboard_url() -> str:
    return f"{CDN}/scoreboard/todaysScoreboard_00.json"


def load_file(path: Path) -> dict:
    return json.loads(path.read_text())
