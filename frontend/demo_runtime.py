"""Standalone adapter: shared BM25 search, a small CPU model, and evidence analysis.

No backend, API credentials, fictional papers, or paid requests. Model loading is lazy.
The request budget is per process, not authentication or a durable per-IP quota.
"""

from collections import deque
from threading import BoundedSemaphore, Lock
import time
from types import SimpleNamespace

import requests

from frontend.evidence_analysis import paper_summary as summary, paper_guidance as guidance, research_plan as plan
from frontend.evidence_analysis import condense
from frontend.small_model import get_model, MODEL_ID

from backend import data_pipeline
from backend.errors import SourceUnavailable
from backend.search_context import contextualize_query
from backend.sources import diagnostics

_search_times = deque()
_budget_lock = Lock()
_search_capacity = BoundedSemaphore(3)


def _admit_search():
    with _budget_lock:
        now = time.monotonic()
        while _search_times and _search_times[0] <= now - 60:
            _search_times.popleft()
        if len(_search_times) >= 40:
            error = requests.HTTPError("Search is busy. Try again in a minute.")
            error.response = SimpleNamespace(text=str(error))
            raise error
        _search_times.append(now)


def preview_metadata():
    return {"provider": "abstract_preview", "model": None, "fallback": False,
            "preview": True, "ranking": "bm25"}


def public_config():
    state = get_model().status()
    model = "flan-t5-small-int8" if state == "ready" else "extractive-evidence-v2"
    return {
        "standalone": True, "llm_provider": "onnx_cpu" if state == "ready" else "abstract_preview",
        "model_state": state, "output_revision": f"small-model-v2-{state}",
        "llm_models": {task: model for task in ("summary", "guidance", "research_plan")},
        "generation_modes": {"fast": {"label": "Small-model draft" if state == "ready" else "Evidence analysis preview", "model": model,
                                       "purpose": "Condensed evidence, method and evaluation extraction, and a paper-linked build plan."}},
        "capabilities": {"ranking": "bm25", "dense": False, "reranker": False, "llm": state == "ready"},
    }


def search(payload):
    if not _search_capacity.acquire(blocking=False):
        error = requests.HTTPError("Search is busy. Try again shortly.")
        error.response = SimpleNamespace(text=str(error))
        raise error
    try:
        return _search(payload)
    finally:
        _search_capacity.release()


def _search(payload):
    query = " ".join(str(payload.get("query", "")).split())[:2000]
    domain = str(payload.get("domain", ""))[:120]
    if not query:
        return {"papers": [], "message": "Enter a research question.", "warnings": []}
    _admit_search()
    try:
        papers = data_pipeline.retrieve_and_rank_papers(
            query, payload.get("categories") or [], domain=domain, lightweight=True,
        )
    except SourceUnavailable:
        papers = []
    state = diagnostics()
    message = "" if papers else "No matching evidence is available. Try another query or search field."
    if not papers and state["warnings"]:
        message = "Academic sources are unavailable. Try again shortly; no sample papers have been substituted."
    return {"papers": papers, "message": message, **state, "domain": domain,
            "search_query": query if data_pipeline.extract_arxiv_id(query)
            else data_pipeline.rewrite_search_query(contextualize_query(query, domain)),
            "capabilities": public_config()["capabilities"]}



class DemoClient:
    def __init__(self):
        self.last_generation = {}

    def get_active_backend(self):
        return "standalone"

    def get_config(self):
        return public_config()

    def get_version(self):
        return {"profile": "Streamlit standalone", **public_config()["capabilities"]}

    def health_check(self):
        return {"status": "ok", "models_ready": get_model().status() == "ready", "backend": "standalone",
                "capabilities": public_config()["capabilities"]}

    def post_json(self, path, payload):
        if path == "/search":
            return search(payload)
        handlers = {"/summarize": ("summary", summary), "/paper-guidance": ("guidance", guidance),
                    "/research-plan": ("research_plan", plan)}
        if path not in handlers:
            raise requests.RequestException("Unsupported standalone operation.")
        field, handler = handlers[path]
        base = handler(payload)
        if path == "/summarize":
            prompt = ("Summarize this research abstract in two concise sentences. Use only the abstract.\n\n"
                      + (payload.get("abstract") or "")[:5000])
        elif path == "/paper-guidance":
            prompt = ("What is the main method proposed in this research abstract? Answer in one concise sentence.\n"
                      "Abstract: " + (payload.get("abstract") or "")[:4500])
        else:
            return self._draft_plan(payload)
        text, meta = self._generate(prompt)
        if text:
            if path == "/summarize":
                base = (f"Small-model summary · {MODEL_ID}; verify against the source.\n\n{text}\n\n"
                        f"Source: {payload.get('url') or payload.get('link') or 'Not supplied'}")
            else:
                base = f"Small-model method summary · verify before implementation.\n\n{text}\n\n{base}"
        return {field: base, "generation": meta}

    def _generate(self, prompt):
        text, reason = get_model().generate(prompt)
        if text:
            import re
            # Reject invented numerical results; this is a limited guard, not a grounding guarantee.
            if not set(re.findall(r"\b\d+(?:\.\d+)?", text)).issubset(re.findall(r"\b\d+(?:\.\d+)?", prompt)):
                text, reason = None, "model_output_unusable"
        if text:
            return text, {"provider": "onnx_cpu", "model": MODEL_ID, "fallback": False, "preview": False}
        message = {
            "model_loading": "The small model is loading. This answer uses extractive evidence analysis.",
            "model_busy": "The small model is busy. This answer uses extractive evidence analysis.",
            "model_disabled": "The small model is disabled. This answer uses extractive evidence analysis.",
        }.get(reason, "The small model could not produce a usable answer. This answer uses extractive evidence analysis.")
        return None, {**preview_metadata(), "fallback": True, "fallback_reason": reason,
                      "fallback_message": message}

    def _draft_plan(self, payload):
        snippets = '\n'.join(f"P{i}: {condense(p.get('abstract', ''), payload.get('query', ''), 65)}"
                             for i, p in enumerate((payload.get('papers') or [])[:4], 1))
        prompt = ("Summarize the approach described in this evidence in two sentences relevant to the question.\n"
                  f"Question: {payload.get('query', '')[:350]}\nEvidence:\n{snippets}")
        text, meta = self._generate(prompt)
        body = plan(payload)
        if text:
            body = body.replace("No AI model is used.", "A small CPU model summarizes the evidence; build steps are evidence-based proposals.")
            body = body.replace('# Key Findings', f'# Key Findings\n\n- Model overview (verify against the papers): {text}\n')
        return {"research_plan": body, "generation": meta}

    def stream_post(self, path, payload):
        self.last_generation = {}
        if path != "/research-plan/stream":
            raise requests.RequestException("Unsupported standalone stream.")
        response = self._draft_plan(payload)
        text = response["research_plan"]
        for start in range(0, len(text), 512):
            yield text[start:start + 512]
        self.last_generation = response["generation"]
