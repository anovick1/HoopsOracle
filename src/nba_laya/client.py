"""HTTP client for Laya (local) and Jev (hosted). Same request either way."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class SystemOneError(RuntimeError):
    pass


def retemper(answer: dict, type_temp: float, question_temp: float) -> dict:
    """Swap the server's per-type temperature for a per-question one.

    The checkpoint stores one temperature per question type; training also fits
    one per question. Probabilities out of the model are softmax(z / T_type), so
    softmax(z / T_q) = normalize(p ** (T_type / T_q)).
    """
    if not question_temp or abs(question_temp - type_temp) < 1e-6:
        return answer
    power = type_temp / question_temp
    probs = answer.get("probabilities")
    if not probs:
        return answer
    scaled = {k: max(v, 1e-9) ** power for k, v in probs.items()}
    total = sum(scaled.values()) or 1.0
    scaled = {k: round(v / total, 4) for k, v in scaled.items()}
    out = dict(answer, probabilities=scaled)
    if answer.get("type") == "noul":
        out["noul"] = scaled.get("A", scaled.get("true", answer.get("noul")))
    elif answer.get("type") == "choice":
        out["choice"] = max(scaled, key=scaled.get)
        out["confidence"] = round(scaled[out["choice"]], 4)
    elif answer.get("type") == "score":
        out["score"] = round(sum(int(k) * v for k, v in scaled.items()), 4)
    return out


class LocalLayaClient:
    """Run a Laya checkpoint in-process (a Colab fine-tune folder, or a Hub id).

    Same `system_one(state, questions)` as the HTTP client. Applies the
    per-question temperatures the training script saved next to the weights.
    """

    def __init__(self, path: str, device: str | None = None):
        import laya

        self.path = path
        self.agent = laya.load(path, device=device)
        self.type_temps = {"choice": 1.0, "score": 1.0, "noul": 1.0}
        self.question_temps: dict[str, float] = {}
        cfg_path = f"{path}/rl_agent_config.json"
        try:
            with open(cfg_path) as handle:
                cfg = json.load(handle)
            temps = cfg.get("temperature")
            if isinstance(temps, list) and len(temps) == 3:
                self.type_temps = dict(zip(("choice", "score", "noul"), temps))
            self.question_temps = cfg.get("temperature_by_question", {}) or {}
        except (OSError, json.JSONDecodeError):
            pass

    def system_one(self, state: dict, questions: dict) -> dict:
        result = self.agent.predict(state, questions)
        answers = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers, dict):
            raise SystemOneError(f"local model returned no answers: {type(result).__name__}")
        out = {}
        for qid, answer in answers.items():
            # noul answers come back with just `noul`; give them a probabilities map first.
            if answer.get("type") == "noul" and "probabilities" not in answer:
                p = float(answer["noul"])
                answer = dict(answer, probabilities={"A": round(p, 4), "B": round(1 - p, 4)})
            t_type = self.type_temps.get(answer.get("type"), 1.0)
            out[qid] = retemper(answer, t_type, self.question_temps.get(qid, t_type))
        return out


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
