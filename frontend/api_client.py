import os
import requests
import time
import logging
from typing import Dict, Any, Generator
from types import SimpleNamespace

logger = logging.getLogger(__name__)

import streamlit as st

class APIClient:
    """
    Centralized API client for Streamlit frontend.
    Uses an HTTP backend when configured, or an in-process demo backend on
    Streamlit Cloud where only one Streamlit process is available.
    """
    def __init__(self):
        # 1. Try OS environment variables
        url = os.getenv("ACADEMICFORGE_BACKEND_URL")
        
        # 2. Try Streamlit Secrets (Community Cloud)
        if not url:
            try:
                url = st.secrets.get("ACADEMICFORGE_BACKEND_URL")
            except Exception:
                pass
                
        self.use_in_process_demo = not bool(url)
        self.base_url = url if url else ""
        
        self.timeout_health = 5
        self.timeout_generation = 240
        self.max_retries = 3

    def get_active_backend(self) -> str:
        if self.use_in_process_demo:
            return "streamlit-cloud-demo"
        return "remote" if "localhost" not in self.base_url and "127.0.0.1" not in self.base_url else "local"

    def health_check(self) -> Dict[str, Any]:
        """Check if backend is healthy and return provider/model info."""
        if self.use_in_process_demo:
            return {
                "status": "ok",
                "backend": self.get_active_backend(),
                "provider": "demo",
                "model": "demo-local-research-planner",
                "models_ready": True,
            }
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout_health)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Health check failed for {self.base_url}: {e}")
            return {"status": "error", "backend": self.get_active_backend(), "message": str(e)}

    def get_config(self) -> Dict[str, Any]:
        if self.use_in_process_demo:
            return self._demo_config()
        response = self._request_with_retry("GET", "/config", timeout=self.timeout_health)
        return response.json()

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_in_process_demo:
            return self._demo_post_json(path, payload)
        response = self._request_with_retry("POST", path, json=payload, timeout=self.timeout_generation)
        return response.json()

    def stream_post(self, path: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        if self.use_in_process_demo:
            yield from self._demo_stream_post(path, payload)
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

    def _demo_config(self) -> Dict[str, Any]:
        return {
            "llm_provider": "demo",
            "llm_models": {
                "default": "demo-local-research-planner",
                "summary": "demo-local-research-planner",
                "summary_deep": "DeepSeek-v4-Pro disabled in Streamlit Cloud demo",
                "guidance": "demo-local-research-planner",
                "guidance_deep": "DeepSeek-v4-Pro disabled in Streamlit Cloud demo",
                "research_plan": "DeepSeek-v4-Pro disabled in Streamlit Cloud demo",
            },
            "streamlit_cloud_demo": True,
            "deep_mode_available": False,
            "generation_modes": {
                "fast": {
                    "label": "Fast Mode (Demo)",
                    "model": "demo-local-research-planner",
                    "purpose": "Live retrieval with lightweight evidence-grounded demo generation.",
                },
                "deep": {
                    "label": "Deep Mode (DeepSeek disabled)",
                    "model": "accounts/fireworks/models/deepseek-v4-pro",
                    "purpose": "Unavailable on this public Streamlit Cloud demo.",
                },
            },
        }

    @staticmethod
    def _paper_namespace(payload: Dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(
            title=payload.get("title", ""),
            authors=payload.get("authors", []),
            abstract=payload.get("abstract", ""),
            link=payload.get("link", ""),
            date=payload.get("date", ""),
            paper_id=payload.get("paper_id", ""),
            source=payload.get("source", "arxiv"),
            url=payload.get("url", ""),
            published=payload.get("published", ""),
            bm25_rank=payload.get("bm25_rank"),
            dense_rank=payload.get("dense_rank"),
            rrf_score=payload.get("rrf_score", 0.0),
            metadata=payload.get("metadata", {}),
            generation_mode="fast",
            model_dump=lambda: {**payload, "generation_mode": "fast"},
        )

    def _demo_post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("generation_mode") == "deep":
            raise requests.RequestException(
                "Deep Mode is disabled in this Streamlit Cloud demo. Use Fast Mode here, "
                "or run the AMD GPU branch for Fireworks/DeepSeek."
            )

        if path == "/search":
            from backend.app import normalize_research_categories
            from backend.data_pipeline import not_enough_relevant_message, retrieve_and_rank_papers

            papers = retrieve_and_rank_papers(
                payload.get("query", ""),
                normalize_research_categories(payload.get("categories", [])),
            )
            fallback_used = any(
                paper.get("metadata", {}).get("demo_relevance_fallback")
                for paper in papers
            )
            if fallback_used:
                message = "Showing the closest live research matches found for this query."
            else:
                message = not_enough_relevant_message() if not papers else ""
            return {"papers": papers, "message": message}

        if path == "/summarize":
            from backend.summarizer import summarize_paper

            return {"summary": summarize_paper(payload, mode="fast")}

        if path == "/paper-guidance":
            from backend.guidance_generator import generate_paper_guidance

            return {"guidance": generate_paper_guidance(payload, mode="fast")}

        if path in {"/paper-guidance/cache-status", "/research-plan/cache-status"}:
            return {"cached": False, "cache": "miss"}

        if path == "/research-plan":
            from backend.research_plan_generator import generate_research_plan

            return {
                "research_plan": generate_research_plan(
                    payload.get("papers", []),
                    payload.get("summaries", []),
                    payload.get("query", ""),
                    mode="fast",
                )
            }

        raise requests.RequestException(f"Unsupported demo endpoint: {path}")

    def _demo_stream_post(self, path: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        if payload.get("generation_mode") == "deep":
            raise requests.RequestException(
                "Deep Mode is disabled in this Streamlit Cloud demo. Use Fast Mode here, "
                "or run the AMD GPU branch for Fireworks/DeepSeek."
            )
        if path == "/research-plan/stream":
            from backend.research_plan_generator import stream_research_plan

            yield from stream_research_plan(
                payload.get("papers", []),
                payload.get("summaries", []),
                payload.get("query", ""),
                mode="fast",
            )
            return
        yield from [self._demo_post_json(path, payload).get("research_plan", "")]
