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
Paper {index}: {paper["title"]}
Authors: {authors}
Date: {paper.get("date") or paper.get("published") or "unknown"}
URL: {paper.get("url") or paper.get("link") or ""}
Core idea: {_truncate(sections.get("core idea") or sections.get("one-sentence takeaway") or sections.get("what problem it solves") or paper.get("abstract", ""), 500)}
Method: {_truncate(sections.get("method") or sections.get("how it works"), 450)}
Implementation relevance: {_truncate(sections.get("implementation notes") or sections.get("why it matters") or sections.get("why a builder should care"), 450)}
Limitations: {_truncate(sections.get("limitations or unknowns") or sections.get("what to verify in the full paper"), 300)}
""".strip()
        )

    return "\n\n".join(paper_blocks)


def generate_roadmap(papers, summaries=None):
    """Generate an implementation roadmap from paper metadata and summaries."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context)
    cached_roadmap = _get_cached_roadmap(cache_key)
    if cached_roadmap:
        return cached_roadmap

    logger.info("Mixed roadmap generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_roadmap_prompt(paper_context)
    roadmap = generate_text(system_prompt, user_prompt, token_budget=1300, task="roadmap")
    ROADMAP_CACHE[cache_key] = roadmap
    cache_set("roadmaps", cache_key, roadmap)
    logger.info("Mixed roadmap generation completed paper_count=%d", len(papers))
    return roadmap


def generate_paper_roadmap(paper):
    """Generate a practical roadmap for understanding and applying one paper."""
    cache_key = paper_roadmap_cache_key(paper)
    cached_roadmap = _get_cached_paper_roadmap(cache_key)
    if cached_roadmap:
        return cached_roadmap

    logger.info("Paper roadmap generation started paper_id=%r", _paper_identity(paper))
    system_prompt, user_prompt = build_paper_roadmap_prompt(paper)
    roadmap = generate_text(system_prompt, user_prompt, token_budget=1100, task="roadmap")
    PAPER_ROADMAP_CACHE[cache_key] = roadmap
    cache_set("paper_roadmaps", cache_key, roadmap)
    logger.info("Paper roadmap generation completed paper_id=%r", _paper_identity(paper))
    return roadmap


def stream_roadmap(papers, summaries=None):
    """Yield roadmap text chunks while generating, then persist the completed roadmap."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context)
    cached_roadmap = _get_cached_roadmap(cache_key)
    if cached_roadmap:
        yield cached_roadmap
        return

    logger.info("Mixed roadmap stream generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_roadmap_prompt(paper_context)
    chunks = []
    raw_chunks = generate_text_stream(
        system_prompt,
        user_prompt,
        token_budget=1300,
        task="roadmap",
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


def build_roadmap_prompt(paper_context):
    system_prompt = (
        "You are AcademicForge, a senior research engineer. Create practical "
        "mixed implementation roadmaps from selected academic papers. Synthesize "
        "across all selected papers instead of producing separate roadmaps. "
        "Prefer small, portable models and designs that run well on local MLX. "
        "Be concrete and engineering-focused. Do not include conversational "
        "prefaces."
    )
    user_prompt = f"""
Use the papers below to create an implementation plan.

{paper_context}

Return one mixed roadmap only. Do not include individual per-paper roadmaps.
Start with a descriptive title in this style:
Implementation Roadmap for <combined project direction>

Then return concise Markdown with these exact numbered sections:
1. Selected paper comparison
2. Recommended direction
3. Goal
4. Minimal working prototype
5. Model/runtime choices
6. Data and preprocessing
7. Core implementation steps
8. Evaluation plan
9. Optimization for low-resource machines
10. Risks and open questions

In "Selected paper comparison", include one short line for every selected paper:
Paper N (paper title): role – why it matters for the combined roadmap.
In "Recommended direction", explicitly say which paper or combination of papers
the builder should go with and why.
In "Core implementation steps", use short labeled subsections when useful.
Make the steps specific enough that a developer can start building.
If the selected papers cover different problem types, such as multimodal
hallucination, token-level text hallucination, RAG hallucination, benchmarks, and
theory, do not collapse them into one paper's method. Choose a coherent MVP and
state what is out of scope for the first version.
Do not include long code blocks. Prefer checklists and short commands.
Do not include unrelated architectures or placeholder examples.
Do not repeat the paper summaries.
""".strip()
    return system_prompt, user_prompt


def build_paper_roadmap_prompt(paper):
    authors = ", ".join(paper.get("authors", []))
    categories = ", ".join(paper.get("metadata", {}).get("categories", []))
    system_prompt = (
        "You are AcademicForge, a senior research mentor. Create practical, "
        "paper-specific learning and implementation roadmaps. Be concrete, "
        "honest about unknowns, and avoid inventing details not supported by "
        "the title, abstract, and metadata. Do not include conversational "
        "prefaces. Prefer plain builder language over abstract-like prose."
    )
    user_prompt = f"""
Paper title: {paper.get("title", "")}
Authors: {authors}
Date: {paper.get("date") or paper.get("published") or "unknown"}
Source: {paper.get("source", "arxiv")}
URL: {paper.get("url") or paper.get("link") or ""}
Categories: {categories}

Abstract:
{_truncate(paper.get("abstract", ""), 2500)}

Return concise Markdown with these exact sections:
1. What this paper is about
2. Key concepts to understand first
3. Required background knowledge
4. Step-by-step reading plan
5. Implementation plan
6. Tools and libraries
7. Possible use cases
8. Difficulty level
9. Estimated learning path
10. How to use this in a project

Make the roadmap practical for a builder deciding whether to implement or use
this paper. If an implementation detail is not present in the abstract, say what
to verify in the full paper instead of guessing.

Rules:
- In section 1, use 2-3 plain-English bullets, not a rewritten abstract.
- Only name benchmarks, datasets, models, algorithms, or libraries if they appear in the title, abstract, or metadata.
- Do not invent training steps, fine-tuning requirements, reward model designs, or evaluation names.
- For missing implementation details, write "Verify in the full paper:" followed by the specific thing to check.
- Keep each section short enough to scan in a demo.
""".strip()
    return system_prompt, user_prompt


def roadmap_cache_key(papers, summaries=None, paper_context=None):
    paper_context = paper_context or build_roadmap_context(papers, summaries)
    return make_cache_key(
        "roadmap-v4",
        model_name("roadmap"),
        [
            paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")
            for paper in papers
        ],
        summaries or [],
        paper_context,
    )


def paper_roadmap_cache_key(paper):
    metadata = paper.get("metadata", {}) or {}
    return make_cache_key(
        "paper-roadmap-v5",
        model_name("roadmap"),
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


def roadmap_cache_status(papers, summaries=None):
    cache_key = roadmap_cache_key(papers, summaries)
    if cache_key in ROADMAP_CACHE:
        return {"cached": True, "cache": "memory"}
    if cache_get("roadmaps", cache_key):
        return {"cached": True, "cache": "disk"}
    return {"cached": False, "cache": "miss"}


def paper_roadmap_cache_status(paper):
    cache_key = paper_roadmap_cache_key(paper)
    if cache_key in PAPER_ROADMAP_CACHE:
        return {"cached": True, "cache": "memory"}
    if cache_get("paper_roadmaps", cache_key):
        return {"cached": True, "cache": "disk"}
    return {"cached": False, "cache": "miss"}
