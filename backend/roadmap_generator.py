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


def generate_roadmap(papers, summaries=None, query=""):
    """Generate an implementation roadmap from paper metadata and summaries."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context, query)
    cached_roadmap = _get_cached_roadmap(cache_key)
    if cached_roadmap:
        return cached_roadmap

    logger.info("Mixed roadmap generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_roadmap_prompt(paper_context, query)
    roadmap = generate_text(system_prompt, user_prompt, token_budget=1900, task="roadmap")
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


def stream_roadmap(papers, summaries=None, query=""):
    """Yield roadmap text chunks while generating, then persist the completed roadmap."""
    paper_context = build_roadmap_context(papers, summaries)
    cache_key = roadmap_cache_key(papers, summaries, paper_context, query)
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
        "You are AcademicForge. Your job is not to summarize papers. Your job "
        "is to help the user solve their research problem by synthesizing "
        "insights across multiple papers and converting research into actionable "
        "implementation guidance. The user question is the primary objective; "
        "the papers are evidence. Do not explain each paper in isolation. Do "
        "not include conversational prefaces."
    )
    user_prompt = f"""
User question:
{query or "No explicit question was provided. Infer the goal from the selected papers."}

Retrieved evidence:

{paper_context}

Create one synthesis for solving the user's problem. Use only retrieved evidence.
Add citations like [1], [2], [3] for every major claim. Citation numbers must
match the Evidence numbers above. If the evidence does not support a claim, say
what needs to be verified instead of guessing.

Return concise Markdown with these exact sections:
1. User Goal Analysis
2. Paper Relevance Analysis
3. Research Synthesis
4. Most Useful Insights
5. Recommended Direction
6. Implementation Roadmap
7. Builder Guidance
8. Evidence Grounding
9. References

Section requirements:

1. User Goal Analysis
- Identify what the user is actually trying to achieve.
- Explain the likely end goal in 2-3 sentences.

2. Paper Relevance Analysis
- For each evidence item, explain why it was retrieved and how it contributes
  to solving the user's problem.
- Assign exactly one category for each item: Foundational, Survey, State of the Art,
  Implementation Focused, Evaluation Focused, Alternative Approach, Contrarian View.
- Do not summarize the whole paper.

3. Research Synthesis
- Include these subsections exactly:
  - What the papers agree on
  - What the papers disagree on
  - Important limitations
  - Research gaps
- This section is mandatory.

4. Most Useful Insights
- Provide the 5 most valuable insights across all evidence.
- Organize by importance, not by paper.

5. Recommended Direction
- Include: Most practical approach, Most promising approach, Most innovative approach,
  and Lowest-cost approach.
- Explain why each direction fits the evidence.

6. Implementation Roadmap
- Create a roadmap for solving the user's problem, not for individual papers.
- Include Phase 1 - Foundation, Phase 2 - Prototype, Phase 3 - Evaluation,
  and Phase 4 - Optimization.
- For each phase include Goals, Deliverables, Recommended tools, and Risks.

7. Builder Guidance
- If the user wants to build something, provide Datasets, Models, Frameworks,
  Evaluation metrics, and Deployment options.
- Recommend only tools supported by the evidence when possible.

8. Evidence Grounding
- List the major claims and the citation IDs supporting each claim.

9. References
- For each citation include:
  [N] Paper Title
  Authors
  Year
  Link
  Supporting evidence: "short excerpt from retrieved text"
- Keep excerpts under 2 sentences.

Critical rules:
- Never output "What this paper is about".
- Never output "Step-by-step reading plan".
- Never output "Difficulty level".
- Never output "Estimated learning path".
- Never output generic AI advice.
- The final output should feel like a senior researcher helping a builder decide
  what to do next, not a paper summary tool.
""".strip()
    return system_prompt, user_prompt


def build_paper_roadmap_prompt(paper):
    authors = ", ".join(paper.get("authors", []))
    categories = ", ".join(paper.get("metadata", {}).get("categories", []))
    system_prompt = (
        "You are AcademicForge. Your job is not to summarize papers. Convert "
        "retrieved evidence into actionable guidance for a builder. Use only "
        "the paper title, abstract, and metadata. Do not include conversational "
        "prefaces and do not invent implementation details."
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

Return concise Markdown with these exact sections:
1. Builder Goal Fit
2. Evidence Contribution
3. Actionable Insight
4. Implementation Direction
5. Evaluation Strategy
6. Risks And Verification Needs
7. Evidence Grounding
8. Reference

Requirements:
- Focus on how this evidence helps a builder decide what to do next.
- Do not summarize the full paper.
- Add citation [1] for every major claim.
- Only name benchmarks, datasets, models, algorithms, or libraries if they appear in the title, abstract, or metadata.
- Do not invent training steps, fine-tuning requirements, reward model designs, or evaluation names.
- If an implementation detail is missing, say "Verify in the full paper:" followed by the specific thing to check.
- Never output "What this paper is about".
- Never output "Step-by-step reading plan".
- Never output "Difficulty level".
- Never output "Estimated learning path".
""".strip()
    return system_prompt, user_prompt


def roadmap_cache_key(papers, summaries=None, paper_context=None, query=""):
    paper_context = paper_context or build_roadmap_context(papers, summaries)
    return make_cache_key(
        "roadmap-v5-synthesis",
        model_name("roadmap"),
        query or "",
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
        "paper-guidance-v1",
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


def roadmap_cache_status(papers, summaries=None, query=""):
    cache_key = roadmap_cache_key(papers, summaries, query=query)
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
