"""Standalone Streamlit Cloud demo for AcademicForge.

This file intentionally avoids the local FastAPI and model-server path. It keeps
the public demo cheap by using live paper retrieval plus deterministic synthesis
unless paid LLM generation is explicitly enabled in Streamlit secrets.
"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_pipeline import (  # noqa: E402
    FINAL_EVIDENCE_TARGET,
    filter_relevant_papers,
    rank_papers,
    rewrite_search_query,
    search_live_candidates,
    select_evidence_set,
)
from backend.sample_data import get_sample_papers  # noqa: E402
from frontend.cloud_usage_limiter import consume_one_request, limiter_salt  # noqa: E402


MAX_QUERY_CHARS = 220
DEFAULT_QUERY = "Reducing hallucinations in retrieval augmented generation"


def secret_or_env(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def secret_flag(name: str, default: bool = False) -> bool:
    value = secret_or_env(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def request_limit() -> int:
    try:
        return max(1, int(secret_or_env("ACADEMICFORGE_CLOUD_REQUEST_LIMIT", "1")))
    except ValueError:
        return 1


def max_llm_tokens() -> int:
    try:
        return min(900, max(250, int(secret_or_env("ACADEMICFORGE_CLOUD_MAX_TOKENS", "650"))))
    except ValueError:
        return 650


def get_client_identifier() -> str:
    """Extract the visitor IP from common proxy headers when Streamlit exposes them."""
    headers = {}
    context = getattr(st, "context", None)
    if context is not None:
        try:
            headers = {str(k).lower(): str(v) for k, v in dict(context.headers).items()}
        except Exception:
            headers = {}

    for header_name in (
        "x-forwarded-for",
        "x-real-ip",
        "cf-connecting-ip",
        "forwarded",
    ):
        value = headers.get(header_name, "").strip()
        if value:
            return value.split(",")[0].replace("for=", "").strip()

    return secret_or_env("ACADEMICFORGE_DEV_CLIENT_ID", "local-dev-client")


def sanitize_query(query: str) -> str:
    return " ".join(query.strip().split())[:MAX_QUERY_CHARS]


@st.cache_data(ttl=900, show_spinner=False)
def retrieve_cloud_papers(query: str) -> tuple[list[dict], str]:
    """Return a small evidence set using free academic-source APIs."""
    search_query = rewrite_search_query(query)
    candidates, direct_id = search_live_candidates(search_query, max_results=18)
    if direct_id:
        return select_evidence_set(candidates, target=1), "live academic source"

    relevant = filter_relevant_papers(query, candidates)
    source_label = "live academic sources"
    if len(relevant) < 3:
        relevant = filter_relevant_papers(query, get_sample_papers())
        source_label = "synthetic fallback data"

    if not relevant:
        return [], source_label

    ranked = rank_papers(query, relevant, top_k=12)
    selected = select_evidence_set(
        ranked,
        target=min(6, FINAL_EVIDENCE_TARGET),
    )
    return selected, source_label


def paper_excerpt(paper: dict, limit: int = 420) -> str:
    text = " ".join((paper.get("abstract") or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def deterministic_summary(paper: dict) -> str:
    title = paper.get("title", "Selected paper")
    category = paper.get("metadata", {}).get("academicforge_category", "evidence")
    return (
        f"**{html.escape(title)}** is treated as {html.escape(category)} evidence. "
        f"{html.escape(paper_excerpt(paper, 320))}"
    )


def deterministic_research_plan(query: str, papers: list[dict]) -> str:
    titles = [paper.get("title", "Untitled paper") for paper in papers[:4]]
    source_mix = sorted({paper.get("source", "unknown") for paper in papers})
    categories = sorted(
        {
            paper.get("metadata", {}).get("academicforge_category", "Uncategorized")
            for paper in papers
        }
    )
    key_findings = "\n".join(
        f"- {paper.get('title', 'Untitled paper')}: {paper_excerpt(paper, 180)}"
        for paper in papers[:4]
    )
    return f"""# Research Plan

## Research Focus

Use the retrieved papers to scope a practical build around: {query}. This public
cloud demo uses deterministic synthesis by default so it does not spend API
credits unless paid generation is explicitly enabled by the owner.

## Evidence Used

- Sources: {", ".join(source_mix) if source_mix else "none"}
- Evidence categories: {", ".join(categories) if categories else "none"}
- Top papers: {", ".join(titles) if titles else "none"}

## Key Findings

{key_findings or "- No relevant papers were retrieved."}

## Recommended Build

Recommended Architecture:
Start with a retrieval-first assistant: collect papers, rank them, select a
small evidence set, and generate recommendations only from that selected
evidence.

Core Components:
Search adapter, lexical ranking, dense-ranking fallback, rank fusion, evidence
selection, report builder, and Markdown export.

Deployment Considerations:
Keep public demos lightweight. Use deterministic summaries by default, cap
optional model tokens, and limit each IP address to one cloud analysis.

Evaluation Strategy:
Track whether expected papers rise above unrelated papers for synthetic queries,
then measure retrieval latency and report generation time separately.

## Limitations

The public cloud version uses abstracts and metadata only. It is rate-limited,
best-effort IP tracking can reset when Streamlit redeploys, and deterministic
mode is not a replacement for full local MLX/ROCm generation.
"""


def stream_text(text: str) -> Iterable[str]:
    for token in text.split(" "):
        yield token + " "
        time.sleep(0.006)


def build_llm_prompt(query: str, papers: list[dict]) -> str:
    evidence = "\n\n".join(
        (
            f"[{index}] {paper.get('title', 'Untitled paper')}\n"
            f"Source: {paper.get('source', 'unknown')}\n"
            f"Abstract: {paper_excerpt(paper, 700)}"
        )
        for index, paper in enumerate(papers[:5], start=1)
    )
    return f"""
User research question:
{query}

Retrieved evidence:
{evidence}

Write a concise Markdown research plan with these sections:
Research Focus, Key Findings, Recommended Build, Evaluation Strategy, Limitations.
Only use the retrieved evidence. Do not invent benchmarks or metrics.
""".strip()


def stream_optional_llm_plan(query: str, papers: list[dict]) -> Iterable[str]:
    api_key = secret_or_env("FIREWORKS_API_KEY")
    if not api_key or not secret_flag("ACADEMICFORGE_CLOUD_ENABLE_LLM", False):
        yield from stream_text(deterministic_research_plan(query, papers))
        return

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=secret_or_env(
                "ACADEMICFORGE_CLOUD_MODEL",
                "accounts/fireworks/models/deepseek-v4-pro",
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are AcademicForge. Produce concise research plans "
                        "from retrieved paper evidence only."
                    ),
                },
                {"role": "user", "content": build_llm_prompt(query, papers)},
            ],
            max_tokens=max_llm_tokens(),
            temperature=0.2,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
    except Exception as exc:
        yield from stream_text(
            deterministic_research_plan(query, papers)
            + f"\n\nLLM generation failed and deterministic mode was used instead: {exc}"
        )


def markdown_report(query: str, papers: list[dict], plan: str, source_label: str) -> str:
    paper_sections = "\n\n".join(
        f"### {index}. {paper.get('title', 'Untitled paper')}\n\n{paper_excerpt(paper, 900)}"
        for index, paper in enumerate(papers, start=1)
    )
    return f"""# AcademicForge Cloud Report

## Query

{query}

## Retrieval Source

{source_label}

## Selected Evidence

{paper_sections}

## Research Plan

{plan}
"""


def render_status() -> None:
    llm_enabled = secret_flag("ACADEMICFORGE_CLOUD_ENABLE_LLM", False) and bool(
        secret_or_env("FIREWORKS_API_KEY")
    )
    mode = "Paid LLM enabled with token cap" if llm_enabled else "No paid LLM calls"
    st.caption(
        f"{mode}. Limit: {request_limit()} cloud analysis per IP. "
        "Raw IP addresses are not stored."
    )


st.set_page_config(page_title="AcademicForge Cloud", page_icon="AF", layout="wide")
st.title("AcademicForge Cloud")
st.subheader("Question -> Papers -> Plan")
render_status()

with st.expander("Public demo controls", expanded=False):
    st.write(
        "This Streamlit Cloud entry point is separate from the local hackathon "
        "MLX/ROCm app. It runs retrieval with lightweight dependencies and uses "
        "deterministic synthesis unless the owner explicitly enables paid LLM "
        "generation in Streamlit secrets."
    )
    st.write(
        "The one-request limit is enforced with a salted hash of the client IP. "
        "It is a cost-control mechanism, not a security boundary."
    )

query = st.text_input(
    "Research question",
    value=DEFAULT_QUERY,
    max_chars=MAX_QUERY_CHARS,
)
run_clicked = st.button("Run one cloud analysis", type="primary")

if run_clicked:
    clean_query = sanitize_query(query)
    if not clean_query:
        st.warning("Enter a research question first.")
        st.stop()

    decision = consume_one_request(
        get_client_identifier(),
        salt=secret_or_env("ACADEMICFORGE_LIMITER_SALT", limiter_salt()),
        request_limit=request_limit(),
    )
    if not decision.allowed:
        st.error(decision.reason)
        st.stop()

    with st.spinner("Retrieving and ranking papers..."):
        papers, source_label = retrieve_cloud_papers(clean_query)

    if not papers:
        st.warning("No relevant papers were found. Try a more specific query.")
        st.stop()

    st.success(f"Retrieved {len(papers)} papers from {source_label}.")
    st.dataframe(
        [
            {
                "Rank": index,
                "Title": paper.get("title", ""),
                "Source": paper.get("source", ""),
                "Category": paper.get("metadata", {}).get(
                    "academicforge_category",
                    "Uncategorized",
                ),
                "BM25": paper.get("bm25_rank"),
                "Dense": paper.get("dense_rank"),
                "RRF": round(float(paper.get("rrf_score", 0.0) or 0.0), 5),
            }
            for index, paper in enumerate(papers, start=1)
        ],
        hide_index=True,
        width="stretch",
    )

    with st.expander("Selected evidence", expanded=True):
        for index, paper in enumerate(papers, start=1):
            st.markdown(f"**{index}. {paper.get('title', 'Untitled paper')}**")
            st.markdown(deterministic_summary(paper))

    st.markdown("## Research Plan")
    plan = st.write_stream(stream_optional_llm_plan(clean_query, papers))
    st.download_button(
        "Download report",
        markdown_report(clean_query, papers, str(plan), source_label),
        file_name="academicforge-cloud-report.md",
        mime="text/markdown",
    )
