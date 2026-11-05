"""Presentation components, independent of backend requests and workflow state."""

import html
import re

import streamlit as st


RESEARCH_PLAN_SECTION_ORDER = [
    "Research Focus",
    "Key Findings",
    "Research Gaps",
    "Recommended Build",
    "Builder Guidance",
    "Selected Evidence",
]

RECOMMENDED_BUILD_FIELDS = [
    "Recommended Architecture",
    "Core Components",
    "Deployment Considerations",
    "Evaluation Strategy",
]

RECOMMENDED_BUILD_LABELS = {
    "Recommended Architecture": "Architecture",
    "Core Components": "Core components",
    "Deployment Considerations": "Deployment considerations",
    "Evaluation Strategy": "Evaluation strategy",
}

RESEARCH_PLAN_SECTION_LABELS = {
    "Research Focus": "Research focus",
    "Key Findings": "Key findings",
    "Research Gaps": "Research gaps",
    "Recommended Build": "Recommended build",
    "Builder Guidance": "Builder guidance",
    "Selected Evidence": "Sources",
}

SECTION_ALIASES = {
    "research insights": "Research Focus",
    "consensus": "Key Findings",
    "evidence used": None,
    "supporting evidence": None,
}


def split_into_sections(text):
    header_map = {header.lower(): header for header in RESEARCH_PLAN_SECTION_ORDER}
    sections = {}
    current, buffer = None, []
    for line in text.splitlines():
        key = line.strip().strip("*#").strip().rstrip(":").lower()
        alias = SECTION_ALIASES.get(key, key)
        if alias is None:
            if current:
                sections[current] = "\n".join(buffer).strip()
            current, buffer = None, []
        elif str(alias).lower() in header_map:
            if current:
                sections[current] = "\n".join(buffer).strip()
            current, buffer = header_map[str(alias).lower()], []
        else:
            if current:
                buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()
    return sections


def split_recommended_build(body):
    field_map = {field.lower(): field for field in RECOMMENDED_BUILD_FIELDS}
    fields, current, buffer = {}, None, []
    for line in body.splitlines():
        match = re.match(r"^\s*([A-Za-z ]+):\s*(.*)$", line)
        label = match.group(1).strip().lower() if match else ""
        if label in field_map:
            if current:
                fields[current] = "\n".join(buffer).strip()
            current = field_map[label]
            buffer = [match.group(2)] if match.group(2) else []
        elif current:
            buffer.append(line)
    if current:
        fields[current] = "\n".join(buffer).strip()
    return fields


def extract_bullets(text):
    # Small models often run bullets together on one line ("* a * b"); split them first.
    text = re.sub(r"(?<=\S)\s+\*\s+(?=\S)", "\n* ", text)
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet_match = re.match(r"^[*\-]\s*(.*)$", stripped)
        numbered_match = re.match(r"^\d+\.\s*(.*)$", stripped)
        if bullet_match:
            items.append(bullet_match.group(1))
        elif numbered_match:
            items.append(numbered_match.group(1))
    return items


def extract_numbered(text):
    items = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+\.\s*(.*)$", line.strip())
        if match:
            items.append(match.group(1))
    return items


def render_list_or_body(text, ordered=False):
    items = extract_numbered(text) if ordered else extract_bullets(text)
    if items:
        tag = "ol" if ordered else "ul"
        rendered_items = "".join(f"<li>{render_citation_spans(item)}</li>" for item in items)
        st.markdown(f"<{tag} class='syn-list'>{rendered_items}</{tag}>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='syn-body'>{render_citation_spans(text)}</div>", unsafe_allow_html=True)


def render_citation_spans(text):
    # Keep each [n] visible as a small marker that points at the Sources list.
    return re.sub(r"\s*\[(?:Evidence\s*)?(\d+)\]", r'<sup class="af-cite">[\1]</sup>', html.escape(text or ""), flags=re.IGNORECASE)


def strip_evidence_citations(text):
    # Only inline citations: a line-leading [n] is a Selected evidence entry and keeps its number.
    return re.sub(r"(?<=\S)[ \t]*\[(?:Evidence\s*)?\d+\]", "", text or "", flags=re.IGNORECASE)


def pulse_loading_html(label):
    return (
        '<div class="af-pulse" role="status"><span class="af-pulse-dots">'
        '<span class="af-pulse-dot"></span><span class="af-pulse-dot"></span><span class="af-pulse-dot"></span>'
        f'</span><span>{html.escape(label)}</span></div>'
    )


def render_research_plan_heading(label):
    st.markdown(f'<div class="syn-heading">{html.escape(label)}</div>', unsafe_allow_html=True)


def render_research_plan(research_plan_text):
    sections = split_into_sections(research_plan_text)
    if not sections:
        st.markdown(research_plan_text)
        return

    for section in RESEARCH_PLAN_SECTION_ORDER:
        body = sections.get(section)
        if not body:
            continue

        render_research_plan_heading(RESEARCH_PLAN_SECTION_LABELS[section])

        if section == "Recommended Build":
            build_fields = split_recommended_build(body)
            for field in RECOMMENDED_BUILD_FIELDS:
                value = build_fields.get(field)
                if not value or field in ("Implementation Difficulty", "Estimated Build Time"):
                    continue
                st.markdown(
                    f'<div class="syn-field-label">{html.escape(RECOMMENDED_BUILD_LABELS[field])}</div>',
                    unsafe_allow_html=True,
                )
                if field in ("Core Components", "Expected Benefits", "Expected Tradeoffs"):
                    render_list_or_body(value)
                else:
                    st.markdown(
                        f'<div class="syn-field-value">{render_citation_spans(value)}</div>',
                        unsafe_allow_html=True,
                    )
            badges = [
                build_fields[field]
                for field in ("Implementation Difficulty", "Estimated Build Time")
                if build_fields.get(field)
            ]
            if badges:
                badge_html = "".join(f'<span class="syn-badge">{html.escape(b)}</span>' for b in badges)
                st.markdown(badge_html, unsafe_allow_html=True)

        elif section == "Builder Guidance":
            render_list_or_body(body, ordered=True)

        elif section == "Selected Evidence":
            st.markdown(body)

        else:
            render_list_or_body(body)


def render_research_plan_panel(research_plan_text):
    with st.container(key="plan_panel"):
        if isinstance(research_plan_text, str):
            render_research_plan(research_plan_text)
        else:
            st.json(research_plan_text)


def paper_insight_panel_html(label, text):
    return (
        f'<div class="af-insight"><div class="af-insight-label">{html.escape(label)}</div>'
        f'<div class="af-insight-text">{html.escape(text)}</div></div>'
    )


def paper_insight_loading_html(label, loading_text):
    return f'<div class="af-insight"><div class="af-insight-label">{html.escape(label)}</div>{pulse_loading_html(loading_text)}</div>'


def render_research_plan_empty_state():
    st.markdown(
        '<div class="af-empty">Pick two to four papers above and write a plan. '
        "It combines them into one build direction with citations back to each paper.</div>",
        unsafe_allow_html=True,
    )


def render_research_plan_loading_state(label="Writing the plan"):
    st.markdown(pulse_loading_html(label), unsafe_allow_html=True)
