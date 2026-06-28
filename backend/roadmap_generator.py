from backend.llm import generate_text


def generate_roadmap(papers, summaries=None):
    """Generate an implementation roadmap from paper metadata and summaries."""
    summaries = summaries or []
    paper_blocks = []
    for index, paper in enumerate(papers, start=1):
        summary = summaries[index - 1] if index - 1 < len(summaries) else ""
        authors = ", ".join(paper.get("authors", []))
        paper_blocks.append(
            f"""
Paper {index}: {paper["title"]}
Authors: {authors}
Date: {paper.get("date", "unknown")}
Link: {paper.get("link", "")}
Abstract: {paper["abstract"]}
Summary: {summary}
""".strip()
        )

    system_prompt = (
        "You are AcademicForge, a senior research engineer. Create practical "
        "implementation roadmaps from academic papers. Prefer small, portable "
        "models and designs that can run across Apple MLX now and AMD/ROCm "
        "later. Be concrete and engineering-focused."
    )
    user_prompt = f"""
Use the papers below to create an implementation plan.

{chr(10).join(paper_blocks)}

Return concise Markdown with these exact sections:
1. Goal
2. Minimal working prototype
3. Model/runtime choices
4. Data and preprocessing
5. Core implementation steps
6. Evaluation plan
7. Optimization for low-resource machines
8. Risks and open questions

Make the steps specific enough that a developer can start building.
Do not include long code blocks. Prefer checklists and short commands.
Do not include unrelated architectures or placeholder examples.
""".strip()

    return generate_text(system_prompt, user_prompt, token_budget=1800)
