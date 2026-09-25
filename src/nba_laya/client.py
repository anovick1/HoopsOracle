"""HTTP client for Laya (local) and Jev (hosted). Same request either way."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class SystemOneError(RuntimeError):
    pass


class SystemOneClient:
    def __init__(self, base_url: str, api_key: str | None = None, model: str = "jev-latest", timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def system_one(self, state: dict, questions: dict) -> dict:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise SystemOneError(f"{exc.code} from {self.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SystemOneError(f"could not reach {self.base_url}: {exc.reason}") from exc
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise SystemOneError(f"response had no answers: {list(payload)[:8]}")
        return answers

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
