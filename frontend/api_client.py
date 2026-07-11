import os
import requests
import time
import logging
from typing import Dict, Any, Generator

logger = logging.getLogger(__name__)

import streamlit as st

class APIClient:
    """
    Centralized API client for Streamlit frontend.
    Handles dynamic routing via VERCEL_API_URL and implements basic retry loops.
    """
    def __init__(self):
        # 1. Try OS environment variables
        url = os.getenv("ACADEMICFORGE_BACKEND_URL") or os.getenv("VERCEL_API_URL")
        
        # 2. Try Streamlit Secrets (Community Cloud)
        if not url:
            try:
                url = st.secrets.get("ACADEMICFORGE_BACKEND_URL") or st.secrets.get("VERCEL_API_URL")
            except Exception:
                pass
                
        # 3. Fallback to the live Vercel Proxy instead of localhost
        self.base_url = url if url else "https://academic-forge-icme.vercel.app"
        
        self.timeout_health = 5
        self.timeout_generation = 240
        self.max_retries = 3

    def get_active_backend(self) -> str:
        return "remote" if "localhost" not in self.base_url and "127.0.0.1" not in self.base_url else "local"

    def health_check(self) -> Dict[str, Any]:
        """Check if backend is healthy and return provider/model info."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout_health)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Health check failed for {self.base_url}: {e}")
            return {"status": "error", "backend": self.get_active_backend(), "message": str(e)}

    def get_config(self) -> Dict[str, Any]:
        response = self._request_with_retry("GET", "/config", timeout=self.timeout_health)
        return response.json()

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request_with_retry("POST", path, json=payload, timeout=self.timeout_generation)
        return response.json()

    def stream_post(self, path: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
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
