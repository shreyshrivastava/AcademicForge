"""Backend client: bounded GET retries, single-attempt generation, typed streams."""

import json
import os
import random
import time

from dotenv import load_dotenv
import requests


class StreamError(requests.RequestException):
    """A stream ended without a complete, usable answer."""


class APIClient:
    def __init__(self):
        load_dotenv(os.getenv("ACADEMICFORGE_ENV_FILE", ".env"))
        self.base_url = os.getenv("ACADEMICFORGE_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
        self.api_key = os.getenv("ACADEMICFORGE_API_KEY", "")
        self.timeout_health = (3, 5)
        self.timeout_generation = (5, 180)
        self.max_retries = 3
        self.last_generation = {}
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key

    def get_active_backend(self):
        return "local" if self.base_url.startswith(("http://localhost:", "http://127.0.0.1:")) else "remote"

    def health_check(self):
        try:
            with self.session.get(f"{self.base_url}/health", timeout=self.timeout_health) as response:
                return response.json()
        except requests.RequestException:
            return {"status": "error", "backend": self.get_active_backend(), "message": "Backend unavailable."}

    def get_config(self):
        with self._request_with_retry("GET", "/config", timeout=self.timeout_health) as response:
            return response.json()

    def get_version(self):
        with self._request_with_retry("GET", "/version", timeout=self.timeout_health) as response:
            return response.json()

    def post_json(self, path, payload):
        # A lost response does not mean generation stopped. Never replay an expensive POST.
        with self._request_with_retry("POST", path, json=payload, timeout=self.timeout_generation) as response:
            return response.json()

    def stream_post(self, path, payload):
        completed = False
        self.last_generation = {}
        with self.session.post(f"{self.base_url}{path}", json=payload, stream=True,
                               timeout=self.timeout_generation,
                               headers={"Accept": "application/x-ndjson"}) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    kind = event["type"]
                    if kind == "error":
                        raise StreamError(event.get("message", "Generation was interrupted."))
                    if kind == "delta":
                        if not isinstance(event.get("text"), str):
                            raise ValueError("Invalid delta")
                        yield event["text"]
                    elif kind == "done":
                        self.last_generation = event.get("generation", {})
                        completed = True
                        break
                    elif kind != "start":
                        raise ValueError("Unknown stream event")
                except (ValueError, KeyError, TypeError) as exc:
                    raise StreamError("Invalid stream response. Start a new attempt.") from exc
        if not completed:
            raise StreamError("Generation was interrupted. Start a new attempt to replace partial output.")

    def _request_with_retry(self, method, path, **kwargs):
        attempts = self.max_retries if method == "GET" else 1
        for attempt in range(attempts):
            try:
                response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == attempts - 1:
                    raise
                time.sleep(min(2, 0.25 * 2 ** attempt + random.uniform(0, 0.1)))
                continue
            if response.status_code in {429, 502, 503, 504} and attempt < attempts - 1:
                try:
                    delay = max(0, float(response.headers.get("Retry-After", "0.5")))
                except ValueError:
                    delay = 0.5
                if delay <= 2:
                    response.close()
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            return response
