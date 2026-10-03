"""NBA official injury reports: fetch the hourly PDFs and parse them.

Reports live at
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{HH_MM}{AM|PM}.pdf
roughly every hour from 11:30 AM to 10:30 PM ET on game days. The NBA stops
serving them after about ten months, so everything read is archived under data/injury/.
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from pathlib import Path

from curl_cffi import requests

BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
TIMES = ["11_30AM", "12_30PM", "01_30PM", "02_30PM", "03_30PM", "04_30PM",
         "05_30PM", "06_30PM", "07_30PM", "08_30PM", "09_30PM", "10_30PM"]
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")


def report_url(day: str, slot: str) -> str:
    return BASE.format(d=day, t=slot)


def fetch_report(day: str, slot: str, out_dir: Path, timeout: float = 20) -> Path | None:
    """Download one report if it exists. Returns the saved path, or None for 403/404."""
    out = out_dir / day / f"{slot}.pdf"
    if out.exists():
        return out
    r = requests.get(report_url(day, slot), impersonate="chrome", timeout=timeout)
    if r.status_code != 200 or "pdf" not in r.headers.get("content-type", ""):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def archive(start: date, end: date, out_dir: Path, pause: float = 0.25, log=print) -> dict[str, int]:
    counts = {"saved": 0, "skipped": 0, "missing": 0}
    day = start
    while day <= end:
        d = day.isoformat()
        for slot in TIMES:
            target = out_dir / d / f"{slot}.pdf"
            if target.exists():
                counts["skipped"] += 1
                continue
            path = fetch_report(d, slot, out_dir)
            counts["saved" if path else "missing"] += 1
            time.sleep(pause)
        if counts["saved"] and counts["saved"] % 100 < len(TIMES):
            log(f"{d}: saved {counts['saved']} so far, missing {counts['missing']}")
        day += timedelta(days=1)
    return counts


# ---- parsing ---------------------------------------------------------------
#
# The PDF is a table. Text extraction scrambles wrapped reason cells (the
# wrapped halves land above and below the player's row), so parsing goes by
# word positions: fixed column x-ranges, rows grouped by y, reason fragments
# attached to the nearest player row.

_HEADER = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2}\s*[AP]M)")
_MATCHUP = re.compile(r"^([A-Z]{3})@([A-Z]{3})$")
COLUMNS = (("date", 0, 118), ("tip", 118, 198), ("matchup", 198, 262), ("team", 262, 423),
           ("player", 423, 583), ("status", 583, 664), ("reason", 664, 10_000))
ROW_TOLERANCE = 2.5
WRAP_TOLERANCE = 12.0


def _column(x0: float) -> str:
    for name, lo, hi in COLUMNS:
        if lo <= x0 < hi:
            return name
    return "reason"


def _lines(words):
    """Group words into visual lines by y, each a dict column -> text."""
    lines: list[tuple[float, dict[str, list[str]]]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        y = w["top"]
        if lines and abs(lines[-1][0] - y) <= ROW_TOLERANCE:
            lines[-1][1].setdefault(_column(w["x0"]), []).append(w["text"])
        else:
            lines.append((y, {_column(w["x0"]): [w["text"]]}))
    return [(y, {k: " ".join(v) for k, v in cols.items()}) for y, cols in lines]


def parse_report(pdf_path: Path) -> list[dict]:
    """One row per (game, team, player, status, reason) in the report."""
    import pdfplumber

    rows: list[dict] = []
    report_time = None
    game: dict | None = None
    team: str | None = None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            m = _HEADER.search(text)
            if m and report_time is None:
                report_time = f"{m.group(1)} {m.group(2)}"
            page_rows: list[tuple[float, dict]] = []
            fragments: list[tuple[float, str]] = []
            for y, cols in _lines(page.extract_words(x_tolerance=1.5, y_tolerance=2)):
                joined = " ".join(cols.values())
                if joined.startswith("Game") or "Injury Report" in joined or joined.startswith("Page "):
                    continue
                if "matchup" in cols and _MATCHUP.match(cols["matchup"]):
                    a, h = _MATCHUP.match(cols["matchup"]).groups()
                    game = {"date": cols.get("date", ""), "tip": cols.get("tip", ""), "away": a, "home": h}
                if "team" in cols:
                    team = cols["team"].replace(" ", "")
                if "player" in cols and cols.get("status") in STATUSES and game:
                    row = {"report_time": report_time, **game, "team": team,
                           "player": cols["player"].replace(" ,", ",").strip(),
                           "status": cols["status"], "reason": cols.get("reason", "")}
                    page_rows.append((y, row))
                elif "reason" in cols and len(cols) == 1:
                    fragments.append((y, cols["reason"]))
            # Wrapped reason text: attach to the nearest player row on the page.
            for fy, text in fragments:
                if not page_rows:
                    continue
                y, row = min(page_rows, key=lambda item: abs(item[0] - fy))
                if abs(y - fy) <= WRAP_TOLERANCE:
                    row["reason"] = (row["reason"] + " " + text).strip() if fy > y else (text + " " + row["reason"]).strip()
            rows.extend(row for _, row in page_rows)
    for row in rows:
        row["reason"] = re.sub(r"\s*-\s*", " - ", row["reason"]).replace(" ;", ";").strip()
    return rows
