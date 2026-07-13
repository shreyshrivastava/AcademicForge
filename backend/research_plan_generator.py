import logging
import re

from backend.llm import generate_text, generate_text_stream, model_name


logger = logging.getLogger(__name__)


def _truncate(text, limit=900):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def strip_evidence_citations(text):
    return re.sub(r"\s*\[(?:Evidence\s*)?\d+\]", "", text or "", flags=re.IGNORECASE)


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


def build_research_plan_context(papers, summaries=None):
    """Create compact paper notes so Research Plan generation stays fast and focused."""
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

def generate_research_plan(papers, summaries=None, query="", model=None, mode=None):
    """Generate a Research Plan based on selected papers and summaries."""
    paper_context = build_research_plan_context(papers, summaries)
    logger.info("Research Plan generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_research_plan_prompt(paper_context, query)
    research_plan = strip_evidence_citations(
        generate_text(system_prompt, user_prompt, token_budget=1900, task="research_plan", model=model)
    )
    logger.info("Research Plan generation completed paper_count=%d", len(papers))
    return research_plan


def stream_research_plan(papers, summaries=None, query="", model=None, mode=None):
    """Yield Research Plan text chunks while generating, then persist the completed Research Plan."""
    paper_context = build_research_plan_context(papers, summaries)
    logger.info("Mixed Research Plan stream generation started paper_count=%d", len(papers))
    system_prompt, user_prompt = build_research_plan_prompt(paper_context, query)
    chunks = []
    raw_chunks = generate_text_stream(
        system_prompt,
        user_prompt,
        token_budget=1900,
        task="research_plan",
        model=model,
    )
    for chunk in _clean_streamed_markdown(raw_chunks):
        chunks.append(chunk)
        yield strip_evidence_citations(chunk)

    research_plan = "".join(chunks).strip()
    logger.info("Mixed Research Plan stream generation completed paper_count=%d", len(papers))


def _clean_streamed_markdown(chunks):
    """Strip common wrapping fences while preserving incremental output."""
    pending = ""
    started = False

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
            elif len(stripped) >= 8:
                started = True

        if started and len(pending) > 16:
            yield pending[:-16]
            pending = pending[-16:]

    if pending:
        yield pending


def build_research_plan_prompt(paper_context, query=""):
    system_prompt = (
        "You are AcademicForge's AI Research Engineer. Your mission is to help "
        "the user move from Question to Papers to Plan. "
        "Analyze all selected papers collectively, propose one coherent build "
        "direction, prioritize architecture, tradeoffs, implementation strategy, "
        "research gaps, and engineering recommendations, and never invent "
        "evidence not present in the selected papers. Do not include "
        "conversational prefaces. You must include every required output heading "
        "exactly once and in the requested order.\n"
        "CRITICAL: You must respond exclusively in English, regardless of the input language."
    )
    user_prompt = f"""
USER GOAL:
{query or "No explicit question was provided."}

RETRIEVED PAPERS:

{paper_context}

INSTRUCTIONS:

1. Base all recommendations and plans strictly on the evidence found in the retrieved papers.
2. Never infer the user's goals or output any User Goal Analysis or User Intent Analysis.
3. Every recommendation (including datasets, frameworks, or architectures) must be traceable to the evidence. Do not suggest anything not present in the papers unless clearly labeled as an example.
4. Keep the 'Research Focus' concise, maximum 150 words. Explain the main technical direction emerging from the literature.
5. Under 'Key Findings', output a maximum of 5 bullets. Each finding must be directly supported by retrieved evidence, but do not include bracketed citation markers.
6. Under 'Research Gaps', output a maximum of 5 bullets focusing on limitations, unanswered questions, and deployment challenges.
7. Under 'Recommended Build', structure your engineering recommendations exactly with these sections:
   - Recommended Architecture
   - Core Components
   - Deployment Considerations
   - Evaluation Strategy
8. Under 'Builder Guidance', provide practical implementation advice. Avoid exact vendor lock-in, exact model recommendations, or exact dataset recommendations unless explicitly supported by evidence.
9. Never expose reasoning, internal thoughts, chain of thought, or reasoning processes. Only output final conclusions.
10. Include every heading below exactly as written. Do not skip 'Research Focus'.

OUTPUT FORMAT:

# Research Plan

# Research Focus

(Concise synthesis, max 150 words, explaining the main technical direction emerging from the literature. Do not use bullets.)

# Key Findings

(Max 5 bullets, each directly supported by retrieved evidence. Do not include citations like [1], [2], or [Evidence 1].)

# Research Gaps

(Max 5 bullets focusing on limitations, unanswered questions, and deployment challenges.)

# Recommended Build

Recommended Architecture:
(Explain the architectural pattern traceable to the papers or labeled as an example.)

Core Components:
(List concrete architecture components from the papers.)

Deployment Considerations:
(Explain deployment considerations supported by the papers.)

Evaluation Strategy:
(Explain evaluation strategy supported by the papers.)

# Builder Guidance

(Provide practical implementation advice. Avoid exact vendor/model/dataset recommendations unless supported by evidence.)
"""
    return system_prompt, user_prompt
