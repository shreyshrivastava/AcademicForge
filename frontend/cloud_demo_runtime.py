"""Cloud-safe local runtime that backs the hosted AcademicForge demo."""

from __future__ import annotations

import time
from typing import Iterable

from backend.data_pipeline import (
    FINAL_EVIDENCE_TARGET,
    filter_relevant_papers,
    rank_papers,
    rewrite_search_query,
    search_live_candidates,
    select_evidence_set,
)
from backend.sample_data import get_sample_papers


def public_config() -> dict:
    return {
        "llm_provider": "cloud_demo",
        "llm_models": {
            "default": "cloud-demo-preview",
            "summary": "evidence-preview",
            "summary_deep": "evidence-preview",
            "guidance": "evidence-preview",
            "guidance_deep": "evidence-preview",
            "research_plan": "demo-research-planner",
        },
        "llm_max_tokens": 650,
        "llm_temperature": 0.0,
        "llm_load_in_4bit": False,
        "generation_modes": {
            "fast": {
                "label": "Fast Mode - cloud demo",
                "model": "cloud-demo-preview",
                "purpose": "Fast preview using live retrieval and deterministic synthesis.",
            },
            "deep": {
                "label": "Deep Mode - cloud demo",
                "model": "demo-research-planner",
                "purpose": "Longer deterministic Research Plan preview for the hosted demo.",
            },
        },
    }


def version_payload() -> dict:
    return {
        "profile": "Streamlit Cloud demo",
        "app_version": "academicforge-cloud-demo",
        "provider": "cloud_demo",
        "accelerator": "none",
        "device": "Streamlit Community Cloud",
        "retrieval_device": "lexical_fallback",
        "llm_provider": "cloud_demo",
        "llm_model": "cloud-demo-preview",
        "summary_model": "evidence-preview",
        "guidance_model": "evidence-preview",
        "research_plan_model": "demo-research-planner",
    }


def retrieve_papers(query: str, categories: list[str] | None = None) -> dict:
    clean_query = " ".join((query or "").split())
    if not clean_query:
        return {"papers": [], "message": "Enter a research question first."}

    source_query = rewrite_search_query(clean_query)
    candidates, direct_id = search_live_candidates(source_query, max_results=28)
    if direct_id:
        return {"papers": select_evidence_set(candidates, target=1), "message": ""}

    relevant = filter_relevant_papers(clean_query, candidates)
    source_label = "live academic sources"
    if len(relevant) < 3:
        relevant = filter_relevant_papers(clean_query, get_sample_papers())
        source_label = "synthetic fallback data"

    if not relevant:
        return {
            "papers": [],
            "message": "No relevant papers found. Try a more specific academic query.",
        }

    ranked = rank_papers(clean_query, relevant, top_k=16)
    focus_categories = [category for category in (categories or []) if category and category != "Balanced"]
    papers = select_evidence_set(
        ranked,
        target=min(8, FINAL_EVIDENCE_TARGET),
        preferred_categories=focus_categories,
    )
    if focus_categories and len(papers) < 3:
        papers = select_evidence_set(ranked, target=min(8, FINAL_EVIDENCE_TARGET))

    for paper in papers:
        metadata = dict(paper.get("metadata", {}) or {})
        metadata["cloud_demo_source"] = source_label
        metadata["cloud_demo_generation"] = "deterministic_preview"
        paper["metadata"] = metadata

    return {"papers": papers, "message": ""}


def paper_excerpt(paper: dict, limit: int = 420) -> str:
    text = " ".join((paper.get("abstract") or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def summarize_paper(paper: dict) -> dict:
    category = paper.get("metadata", {}).get("academicforge_category", "evidence")
    title = paper.get("title", "Selected paper")
    source = paper.get("source", "source")
    summary = f"""What this paper contributes

{title} is treated as {category} evidence from {source}. The abstract indicates
this paper is relevant because: {paper_excerpt(paper, 380)}

How it fits the research plan

Use it as supporting evidence when deciding what problem framing, retrieval
method, evaluation approach, or implementation tradeoff should be carried into
the project plan.
"""
    return {"summary": summary}


def generate_paper_guidance(paper: dict) -> dict:
    category = paper.get("metadata", {}).get("academicforge_category", "Uncategorized")
    title = paper.get("title", "this paper")
    if category == "Implementation Focused":
        use_case = "extract architecture choices, components, and deployment constraints"
    elif category == "Evaluation Focused":
        use_case = "extract metrics, baselines, datasets, and failure cases"
    elif category == "Survey":
        use_case = "map the problem space and identify major families of approaches"
    elif category == "Contrarian View":
        use_case = "capture limitations, failure modes, and assumptions to verify"
    else:
        use_case = "ground the research direction and identify what to verify next"

    guidance = f"""Why this paper matters

{title} is useful as {category.lower()} evidence for the selected question.

How to use it in a project

Use it to {use_case}. The hosted demo derives this preview from the paper title,
abstract, source, and retrieval metadata. The full local app uses the configured
model path for richer paper guidance.

What to verify

Read the full paper before making implementation decisions. Confirm the
experiments, datasets, limitations, and whether any named tools or benchmarks
actually apply to your use case.
"""
    return {"guidance": guidance}


def research_plan_text(papers: list[dict], summaries: list[str] | None = None, query: str = "") -> str:
    selected = papers[:5]
    categories = sorted(
        {
            paper.get("metadata", {}).get("academicforge_category", "Uncategorized")
            for paper in selected
        }
    )
    sources = sorted({paper.get("source", "unknown") for paper in selected})
    findings = "\n".join(
        f"- {paper.get('title', 'Untitled paper')}: {paper_excerpt(paper, 190)}"
        for paper in selected
    )
    summary_context = "\n".join(f"- {summary[:220]}" for summary in (summaries or []) if summary)

    return f"""# Research Plan

# Research Focus

Use the selected evidence to scope a practical build around: {query or "the selected research question"}.
This hosted demo uses the original AcademicForge interface with a cloud-safe
deterministic runtime. It shows the intended retrieval-to-plan workflow without
running the full local MLX/ROCm or model/API backend.

# Key Findings

{findings or "- No selected evidence was provided."}

# Research Gaps

- Validate whether the retrieved papers contain implementation details beyond the abstracts.
- Confirm benchmark definitions, datasets, and assumptions in the full papers.
- Test whether selected retrieval methods generalize to the target domain.
- Measure latency, grounding quality, and failure cases in the full local app.

# Recommended Build

Recommended Architecture:
Use a retrieval-first assistant with source search, ranking, evidence selection,
paper-level summaries, and a final synthesis step.

Core Components:
Live academic source adapters, BM25 ranking, dense-ranking fallback, rank fusion,
paper guidance, selected evidence management, and Markdown export.

Deployment Considerations:
Use the local app for full MLX/ROCm Gemma or optional Fireworks synthesis. Keep
the hosted demo lightweight and IP-limited so it remains stable on Streamlit
Community Cloud.

Evaluation Strategy:
Track ranking consistency for known queries, citation/source quality, answer
grounding, generation latency, and whether selected papers support the final
recommendations.

# Builder Guidance

Selected evidence categories: {", ".join(categories) if categories else "none"}.
Sources: {", ".join(sources) if sources else "none"}.
{summary_context}
"""


def stream_research_plan(
    papers: list[dict],
    summaries: list[str] | None = None,
    query: str = "",
) -> Iterable[str]:
    text = research_plan_text(papers, summaries, query)
    for token in text.split(" "):
        yield token + " "
        time.sleep(0.004)
