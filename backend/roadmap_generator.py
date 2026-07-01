from backend.cache import cache_get, cache_set, make_cache_key
from backend.llm import generate_text
from backend.llm import generate_text_stream
from backend.llm import model_name


ROADMAP_CACHE = {}


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
Retrieval: BM25={paper.get("bm25_rank")}, dense={paper.get("dense_rank")}, RRF={paper.get("rrf_score", 0.0):.5f}
Core idea: {_truncate(sections.get("core idea") or paper.get("abstract", ""), 500)}
Method: {_truncate(sections.get("method"), 450)}
Implementation relevance: {_truncate(sections.get("implementation notes") or sections.get("why it matters"), 450)}
Limitations: {_truncate(sections.get("limitations or unknowns"), 300)}
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

    system_prompt, user_prompt = build_roadmap_prompt(paper_context)
    roadmap = generate_text(system_prompt, user_prompt, token_budget=1300, task="roadmap")
    ROADMAP_CACHE[cache_key] = roadmap
    cache_set("roadmaps", cache_key, roadmap)
    return roadmap


def stream_roadmap(papers, summaries=None):
    """Yield roadmap text chunks while generating, then persist the completed roadmap."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context)
    cached_roadmap = _get_cached_roadmap(cache_key)
    if cached_roadmap:
        yield cached_roadmap
        return

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


def _get_cached_roadmap(cache_key):
    if cache_key in ROADMAP_CACHE:
        return ROADMAP_CACHE[cache_key]

    cached_roadmap = cache_get("roadmaps", cache_key)
    if cached_roadmap:
        ROADMAP_CACHE[cache_key] = cached_roadmap
    return cached_roadmap


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
        "implementation roadmaps from academic papers. Synthesize across all "
        "selected papers instead of anchoring on the first paper. Prefer small, "
        "portable models and designs that run well on local MLX. Be concrete "
        "and engineering-focused."
    )
    user_prompt = f"""
Use the papers below to create an implementation plan.

{paper_context}

Return concise Markdown with these exact sections:
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

In "Selected paper comparison", include one short bullet for every selected paper
and explain what role it should play: primary method, benchmark, theory, detector,
or supporting background.
In "Recommended direction", explicitly say which paper or combination of papers
the builder should go with and why.
Make the steps specific enough that a developer can start building.
If the selected papers cover different problem types, such as multimodal
hallucination, token-level text hallucination, RAG hallucination, benchmarks, and
theory, do not collapse them into one paper's method. Choose a coherent MVP and
state what is out of scope for the first version.
Do not include long code blocks. Prefer checklists and short commands.
Do not include unrelated architectures or placeholder examples.
""".strip()
    return system_prompt, user_prompt


def roadmap_cache_key(papers, summaries=None, paper_context=None):
    paper_context = paper_context or build_roadmap_context(papers, summaries)
    return make_cache_key(
        "roadmap-v1",
        model_name("roadmap"),
        [
            paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")
            for paper in papers
        ],
        summaries or [],
        paper_context,
    )


def roadmap_cache_status(papers, summaries=None):
    cache_key = roadmap_cache_key(papers, summaries)
    if cache_key in ROADMAP_CACHE:
        return {"cached": True, "cache": "memory"}
    if cache_get("roadmaps", cache_key):
        return {"cached": True, "cache": "disk"}
    return {"cached": False, "cache": "miss"}
