import logging

from backend.cache import cache_get, cache_set, make_cache_key
from backend.llm import generate_text
from backend.llm import generate_text_stream
from backend.llm import model_name


ROADMAP_CACHE = {}
PAPER_ROADMAP_CACHE = {}
logger = logging.getLogger(__name__)


def _truncate(text, limit=900):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _extract_summary_sections(summary):
    section_names = {
        "core idea",
        "method",
        "why it matters",
        "implementation notes",
        "limitations or unknowns",
        "one-sentence takeaway",
        "what problem it solves",
        "how it works",
        "why a builder should care",
        "what to verify in the full paper",
    }
    sections = {}
    current_section = None
    current_lines = []

    for raw_line in (summary or "").splitlines():
        line = raw_line.strip()
        normalized = line.rstrip(":").lower()
        if normalized in section_names:
            if current_section:
                sections[current_section] = " ".join(current_lines).strip()
            current_section = normalized
            current_lines = []
        elif current_section and line:
            current_lines.append(line.lstrip("- ").strip())

    if current_section:
        sections[current_section] = " ".join(current_lines).strip()

    return sections


def build_roadmap_context(papers, summaries=None):
    """Create compact paper notes so roadmap generation stays fast and focused."""
    summaries = summaries or []
    paper_blocks = []
    for index, paper in enumerate(papers, start=1):
        summary = summaries[index - 1] if index - 1 < len(summaries) else ""
        sections = _extract_summary_sections(summary)
        authors = ", ".join(paper.get("authors", [])[:4])
        if len(paper.get("authors", [])) > 4:
            authors += ", et al."

        paper_blocks.append(
            f"""
[Evidence {index}] {paper["title"]}
Authors: {authors}
Date: {paper.get("date") or paper.get("published") or "unknown"}
URL: {paper.get("url") or paper.get("link") or ""}
Evidence category: {paper.get("metadata", {}).get("academicforge_category", "Uncategorized")}
Retrieved text: {_truncate(paper.get("abstract", ""), 750)}
Prior extracted notes: {_truncate((sections.get("core idea") or sections.get("one-sentence takeaway") or sections.get("what problem it solves") or "") + " " + (sections.get("method") or sections.get("how it works") or "") + " " + (sections.get("implementation notes") or sections.get("why it matters") or sections.get("why a builder should care") or ""), 650)}
Limitations or verification needs: {_truncate(sections.get("limitations or unknowns") or sections.get("what to verify in the full paper"), 300)}
""".strip()
        )

    return "\n\n".join(paper_blocks)


def generate_roadmap(papers, summaries=None, query="", model=None):
    """Generate an implementation roadmap from paper metadata and summaries."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context, query, model=model)
    cached_roadmap = _get_cached_roadmap(cache_key)
    if cached_roadmap:
        return cached_roadmap

    logger.info("Mixed roadmap generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_roadmap_prompt(paper_context, query)
    roadmap = generate_text(system_prompt, user_prompt, token_budget=1900, task="roadmap", model=model)
    ROADMAP_CACHE[cache_key] = roadmap
    cache_set("roadmaps", cache_key, roadmap)
    logger.info("Mixed roadmap generation completed paper_count=%d", len(papers))
    return roadmap


def generate_paper_roadmap(paper, model=None):
    """Generate a practical roadmap for understanding and applying one paper."""
    cache_key = paper_roadmap_cache_key(paper, model=model)
    cached_roadmap = _get_cached_paper_roadmap(cache_key)
    if cached_roadmap:
        return cached_roadmap

    logger.info("Paper roadmap generation started paper_id=%r", _paper_identity(paper))
    system_prompt, user_prompt = build_paper_roadmap_prompt(paper)
    roadmap = generate_text(system_prompt, user_prompt, token_budget=1100, task="roadmap", model=model)
    PAPER_ROADMAP_CACHE[cache_key] = roadmap
    cache_set("paper_roadmaps", cache_key, roadmap)
    logger.info("Paper roadmap generation completed paper_id=%r", _paper_identity(paper))
    return roadmap


def stream_roadmap(papers, summaries=None, query="", model=None):
    """Yield roadmap text chunks while generating, then persist the completed roadmap."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context, query, model=model)
    cached_roadmap = _get_cached_roadmap(cache_key)
    if cached_roadmap:
        yield cached_roadmap
        return

    logger.info("Mixed roadmap stream generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_roadmap_prompt(paper_context, query)
    chunks = []
    raw_chunks = generate_text_stream(
        system_prompt,
        user_prompt,
        token_budget=1900,
        task="roadmap",
        model=model,
    )
    for chunk in _clean_streamed_markdown(raw_chunks):
        chunks.append(chunk)
        yield chunk

    roadmap = "".join(chunks).strip()
    ROADMAP_CACHE[cache_key] = roadmap
    cache_set("roadmaps", cache_key, roadmap)
    logger.info("Mixed roadmap stream generation completed paper_count=%d", len(papers))


def _get_cached_roadmap(cache_key):
    if cache_key in ROADMAP_CACHE:
        logger.info("Mixed roadmap cache hit cache=memory")
        return ROADMAP_CACHE[cache_key]

    cached_roadmap = cache_get("roadmaps", cache_key)
    if cached_roadmap:
        logger.info("Mixed roadmap cache hit cache=disk")
        ROADMAP_CACHE[cache_key] = cached_roadmap
    else:
        logger.info("Mixed roadmap cache miss")
    return cached_roadmap


def _get_cached_paper_roadmap(cache_key):
    if cache_key in PAPER_ROADMAP_CACHE:
        logger.info("Paper roadmap cache hit cache=memory")
        return PAPER_ROADMAP_CACHE[cache_key]

    cached_roadmap = cache_get("paper_roadmaps", cache_key)
    if cached_roadmap:
        logger.info("Paper roadmap cache hit cache=disk")
        PAPER_ROADMAP_CACHE[cache_key] = cached_roadmap
    else:
        logger.info("Paper roadmap cache miss")
    return cached_roadmap


def _paper_identity(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")


def _clean_streamed_markdown(chunks):
    """Strip common wrapping fences while preserving incremental output."""
    pending = ""
    started = False
    tail_size = 16

    for chunk in chunks:
        pending += chunk
        if not started:
            stripped = pending.lstrip()
            if stripped.startswith("```"):
                newline_index = stripped.find("\n")
                if newline_index == -1:
                    continue
                pending = stripped[newline_index + 1:]
            started = True

        if len(pending) > tail_size:
            yield pending[:-tail_size]
            pending = pending[-tail_size:]

    closing_index = pending.rfind("```")
    if closing_index != -1 and pending[closing_index:].strip() == "```":
        pending = pending[:closing_index].rstrip()

    if pending:
        yield pending


def build_roadmap_prompt(paper_context, query=""):
    system_prompt = (
        "You are AcademicForge's AI Research Engineer. Your mission is not to "
        "summarize papers or write a literature review. Your mission is to help "
        "the user move from Question to Research to Decision to Prototype. "
        "Analyze all selected papers collectively, propose one coherent build "
        "direction, prioritize architecture, tradeoffs, implementation strategy, "
        "research gaps, and engineering recommendations, and never invent "
        "evidence not present in the selected papers. Do not include "
        "conversational prefaces."
    )
    user_prompt = f"""
USER GOAL:
{query or "No explicit question was provided. Infer the goal from the selected papers."}

RETRIEVED PAPERS:

{paper_context}

INSTRUCTIONS:

1. Infer the user's real engineering objective. Do not simply repeat the query.
2. Analyze all papers collectively rather than individually.
3. Extract transferable engineering knowledge:
   - common themes across selected papers
   - frequently occurring approaches
   - implementation patterns
   - limitations and research gaps
   - how the selected research mode should influence the plan
4. Determine the approach that has the strongest support from the selected papers.
5. Recommend one practical implementation that a builder, engineer, researcher,
   student, or startup founder could realistically create.
6. Prioritize actionable recommendations over academic discussion.
7. Ground all conclusions in the provided papers.
8. Never invent evidence not present in the literature.
9. Add citations like [1], [2], [3] for major recommendations and claims.
   Citation numbers must match the Evidence numbers above.
10. If the evidence does not support a recommendation, say what needs to be
    verified instead of guessing.
11. Do not repeat paper summaries. Assume the user has already read Summary and
    Guidance panels.
12. Do not output Evidence Used, Supporting Evidence, or a separate evidence
    grounding section.

OUTPUT FORMAT:

# Research Plan

# User Goal Analysis

2-3 sentences explaining the real engineering objective. Do not simply repeat
the query.

# Research Focus

Concise synthesis of common themes, frequently occurring approaches,
implementation patterns, relevant evidence, and how the selected research mode
influences the plan. Do not use bullet dumps.

# Key Findings

Transferable engineering insights from the selected papers. Avoid obvious
statements and extract what matters for system design.

# Research Gaps

What remains unresolved, weakly supported, or risky.

# Recommended Build

Project Name:

Objective:

Recommended Architecture:

Core Components:
List concrete architecture components. Each component must include a short
explanation. Avoid generic names such as Processing Engine, Monitoring Module,
or Optimization Component.

Implementation Difficulty:
(Beginner / Intermediate / Advanced)

Estimated Build Time:

Why This Approach:
Explain why the evidence supports this recommendation.

Expected Benefits:

Expected Tradeoffs:

# Builder Guidance

Explain recommended implementation order, risks, tradeoffs, validation strategy,
and next steps.

Critical rules:
- Do not output a paper-by-paper summary.
- Do not output a literature review.
- Do not output Evidence Used.
- Do not output Supporting Evidence.
- Do not output "What this paper is about".
- Do not output "Step-by-step reading plan".
- Do not output "Estimated learning path".
- Do not output generic AI advice.
- Keep the answer concise enough for a builder to act on immediately.
""".strip()
    return system_prompt, user_prompt


def build_paper_roadmap_prompt(paper):
    authors = ", ".join(paper.get("authors", []))
    categories = ", ".join(paper.get("metadata", {}).get("categories", []))
    system_prompt = (
        "You are AcademicForge. Your job is to give short, practical guidance "
        "for how a builder can use one paper. Guidance answers only: how can I "
        "use this paper? Use only the paper title, abstract, and metadata. Do "
        "not include conversational prefaces and do not invent implementation "
        "details."
    )
    user_prompt = f"""
Evidence [1]
Paper title: {paper.get("title", "")}
Authors: {authors}
Date: {paper.get("date") or paper.get("published") or "unknown"}
Source: {paper.get("source", "arxiv")}
URL: {paper.get("url") or paper.get("link") or ""}
Categories: {categories}

Abstract:
{_truncate(paper.get("abstract", ""), 2500)}

Requirements:
- Return exactly three short paragraphs.
- Paragraph 1: why this paper matters.
- Paragraph 2: how to use it in a project.
- Paragraph 3: what to watch out for or verify.
- Do not use headings.
- Do not use bullet lists.
- Do not summarize the full paper.
- Only name benchmarks, datasets, models, algorithms, or libraries if they appear in the title, abstract, or metadata.
- Do not invent training steps, fine-tuning requirements, reward model designs, or evaluation names.
- If an implementation detail is missing, say briefly what to verify in the full paper.
- Never output "What this paper is about".
- Never output "Step-by-step reading plan".
- Never output "Difficulty level".
- Never output "Estimated learning path".
""".strip()
    return system_prompt, user_prompt


def roadmap_cache_key(papers, summaries=None, paper_context=None, query="", model=None):
    paper_context = paper_context or build_roadmap_context(papers, summaries)
    return make_cache_key(
        "roadmap-v7-ai-research-engineer-plan",
        model or model_name("roadmap"),
        query or "",
        [
            paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")
            for paper in papers
        ],
        summaries or [],
        paper_context,
    )


def paper_roadmap_cache_key(paper, model=None):
    metadata = paper.get("metadata", {}) or {}
    return make_cache_key(
        "paper-guidance-v2-short-practical",
        model or model_name("roadmap"),
        paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title"),
        paper.get("title"),
        paper.get("abstract"),
        paper.get("authors", []),
        paper.get("date") or paper.get("published"),
        paper.get("source"),
        paper.get("url") or paper.get("link"),
        metadata.get("categories") or [],
        metadata.get("primary_category"),
    )


def roadmap_cache_status(papers, summaries=None, query="", model=None):
    cache_key = roadmap_cache_key(papers, summaries, query=query, model=model)
    if cache_key in ROADMAP_CACHE:
        return {"cached": True, "cache": "memory"}
    if cache_get("roadmaps", cache_key):
        return {"cached": True, "cache": "disk"}
    return {"cached": False, "cache": "miss"}


def paper_roadmap_cache_status(paper, model=None):
    cache_key = paper_roadmap_cache_key(paper, model=model)
    if cache_key in PAPER_ROADMAP_CACHE:
        return {"cached": True, "cache": "memory"}
    if cache_get("paper_roadmaps", cache_key):
        return {"cached": True, "cache": "disk"}
    return {"cached": False, "cache": "miss"}
