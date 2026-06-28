from backend.llm import generate_text


def summarize_paper(paper):
    """Generate a concise, useful research summary for one paper."""
    title = paper["title"]
    authors = ", ".join(paper.get("authors", []))
    abstract = paper["abstract"]

    system_prompt = (
        "You are AcademicForge, a careful academic research assistant. "
        "Summarize papers for builders who need to understand the idea, "
        "method, implementation implications, and limitations. Be specific, "
        "avoid hype, and do not invent details not supported by the abstract."
    )
    user_prompt = f"""
Paper title: {title}
Authors: {authors}
Abstract:
{abstract}

Write a structured summary with these sections:
- Core idea
- Method
- Why it matters
- Implementation notes
- Limitations or unknowns

Keep it concise but useful.
""".strip()

    return generate_text(system_prompt, user_prompt, token_budget=650)
