import os
import requests
import time
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, Generator

logger = logging.getLogger(__name__)

import streamlit as st

from frontend.cloud_usage_limiter import consume_one_request, limiter_salt
from frontend import cloud_demo_runtime

class APIClient:
    """
    Centralized API client for Streamlit frontend.
    Handles local backend routing and implements basic retry loops.
    """
    def __init__(self):
        self.frontend_mode = os.getenv("ACADEMICFORGE_FRONTEND_MODE", "").strip().lower()
        try:
            self.frontend_mode = st.secrets.get("ACADEMICFORGE_FRONTEND_MODE", self.frontend_mode)
        except Exception:
            pass

        # 1. Try OS environment variables
        url = os.getenv("ACADEMICFORGE_BACKEND_URL")
        
        # 2. Try Streamlit Secrets (Community Cloud)
        if not url:
            try:
                url = st.secrets.get("ACADEMICFORGE_BACKEND_URL")
            except Exception:
                pass
                
        # 3. Fallback to the local FastAPI backend.
        self.base_url = (url or "http://127.0.0.1:8000").rstrip("/")
        
        self.timeout_health = 5
        self.timeout_generation = 240
        self.max_retries = 3

    def get_active_backend(self) -> str:
        if self.frontend_mode == "cloud_demo":
            return "cloud_demo"
        return "remote" if "localhost" not in self.base_url and "127.0.0.1" not in self.base_url else "local"

    def health_check(self) -> Dict[str, Any]:
        """Check if backend is healthy and return provider/model info."""
        if self.frontend_mode == "cloud_demo":
            return {
                "status": "ok",
                "backend": "cloud_demo",
                "provider": "cloud_demo",
                "model": "demo-research-planner",
                "models_ready": True,
                "runtime": cloud_demo_runtime.version_payload(),
            }
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout_health)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Health check failed for {self.base_url}: {e}")
            return {"status": "error", "backend": self.get_active_backend(), "message": str(e)}

    def get_config(self) -> Dict[str, Any]:
        if self.frontend_mode == "cloud_demo":
            return cloud_demo_runtime.public_config()
        response = self._request_with_retry("GET", "/config", timeout=self.timeout_health)
        return response.json()

    def get_version(self) -> Dict[str, Any]:
        if self.frontend_mode == "cloud_demo":
            return cloud_demo_runtime.version_payload()
        response = self._request_with_retry("GET", "/version", timeout=self.timeout_health)
        return response.json()

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.frontend_mode == "cloud_demo":
            return self._cloud_demo_post(path, payload)
        response = self._request_with_retry("POST", path, json=payload, timeout=self.timeout_generation)
        return response.json()

    def stream_post(self, path: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        if self.frontend_mode == "cloud_demo":
            yield from self._cloud_demo_stream(path, payload)
            return
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, stream=True, timeout=self.timeout_generation)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
                return
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == self.max_retries - 1:
                    raise e
                logger.warning(f"Stream request failed ({e}). Retrying {attempt + 1}/{self.max_retries}...")
                time.sleep(2 ** attempt)

    def _request_with_retry(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == self.max_retries - 1:
                    raise e
                logger.warning(f"Request failed ({e}). Retrying {attempt + 1}/{self.max_retries}...")
                time.sleep(2 ** attempt)

    def _cloud_demo_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if path == "/search":
            return cloud_demo_runtime.retrieve_papers(
                payload.get("query", ""),
                payload.get("categories") or [],
            )
        if path == "/summarize":
            return cloud_demo_runtime.summarize_paper(payload)
        if path == "/paper-guidance":
            return cloud_demo_runtime.generate_paper_guidance(payload)
        if path == "/research-plan":
            self._consume_cloud_demo_request()
            return {
                "research_plan": cloud_demo_runtime.research_plan_text(
                    payload.get("papers") or [],
                    payload.get("summaries") or [],
                    payload.get("query", ""),
                )
            }
        raise self._http_error(f"Unsupported cloud demo endpoint: {path}")

    def _cloud_demo_stream(self, path: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        if path != "/research-plan/stream":
            raise self._http_error(f"Unsupported cloud demo endpoint: {path}")
        self._consume_cloud_demo_request()
        yield from cloud_demo_runtime.stream_research_plan(
            payload.get("papers") or [],
            payload.get("summaries") or [],
            payload.get("query", ""),
        )

    def _consume_cloud_demo_request(self) -> None:
        try:
            request_limit = int(self._secret_or_env("ACADEMICFORGE_CLOUD_REQUEST_LIMIT", "1"))
        except ValueError:
            logger.warning("Invalid ACADEMICFORGE_CLOUD_REQUEST_LIMIT; defaulting to 1")
            request_limit = 1

        decision = consume_one_request(
            self._client_identifier(),
            salt=self._secret_or_env("ACADEMICFORGE_LIMITER_SALT", limiter_salt()),
            db_path=Path(".academicforge_cloud/usage.sqlite3"),
            request_limit=request_limit,
        )
        if not decision.allowed:
            raise self._http_error(decision.reason)

    def _client_identifier(self) -> str:
        headers = {}
        context = getattr(st, "context", None)
        if context is not None:
            try:
                headers = {str(k).lower(): str(v) for k, v in dict(context.headers).items()}
            except Exception:
                headers = {}
        for header_name in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip", "forwarded"):
            value = headers.get(header_name, "").strip()
            if value:
                return value.split(",")[0].replace("for=", "").strip()
        return self._secret_or_env("ACADEMICFORGE_DEV_CLIENT_ID", "local-dev-client")

    def _secret_or_env(self, name: str, default: str = "") -> str:
        try:
            value = st.secrets.get(name)
            if value is not None:
                return str(value)
        except Exception:
            pass
        return os.getenv(name, default)

    def _http_error(self, message: str) -> requests.HTTPError:
        error = requests.HTTPError(message)
        error.response = SimpleNamespace(text=message)
        return error
