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
    return html.escape(strip_evidence_citations(text))


def strip_evidence_citations(text):
    return re.sub(r"\s*\[(?:Evidence\s*)?\d+\]", "", text or "", flags=re.IGNORECASE)


def pulse_loading_html(label):
    return f"""
    <div class="af-pulse-wrap">
        <div class="af-pulse-dots">
            <div class="af-pulse-dot"></div>
            <div class="af-pulse-dot"></div>
            <div class="af-pulse-dot"></div>
        </div>
        <div class="af-pulse-label">{html.escape(label)}</div>
    </div>
    """


def render_research_plan_heading(label, top=False):
    if top:
        st.markdown(
            f"""
            <div style="font-family: var(--af-font); font-size: 1.35rem; font-weight: 800; color: var(--af-text-hi); margin-bottom: 1.2rem; border-bottom: 2px solid var(--af-border); padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                {html.escape(label)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="syn-heading" style="margin-top: 1.6rem;">
                <span class="syn-marker"></span>
                <span class="syn-heading-label">{html.escape(label)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_research_plan(research_plan_text):
    sections = split_into_sections(research_plan_text)
    if not sections:
        st.markdown(research_plan_text)
        return

    render_research_plan_heading("Research Plan", top=True)

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

        else:
            st.markdown(f"<div class='syn-body'>{render_citation_spans(body)}</div>", unsafe_allow_html=True)


def render_research_plan_panel(research_plan_text):
    with st.container(border=True):
        if isinstance(research_plan_text, str):
            render_research_plan(research_plan_text)
        else:
            st.json(research_plan_text)


def paper_insight_panel_html(label, text):
    return f"""
    <div class="af-insight-block">
        <div class="af-insight-label">{html.escape(label)}</div>
        <div class="af-insight-text">{html.escape(text)}</div>
    </div>
    """


def paper_insight_loading_html(label, loading_text):
    return f"""
    <div class="af-insight-block">
        <div class="af-insight-label">{html.escape(label)}</div>
        {pulse_loading_html(loading_text)}
    </div>
    """


def render_research_plan_empty_state():
    st.markdown(
        """
        <div class="research-plan-stage-empty">
            <div class="research-plan-stage-title">Research Plan will appear here</div>
            <div class="research-plan-stage-copy">
                Select 2-4 papers and generate one combined research-to-prototype plan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_plan_loading_state(label="Generating Research Plan..."):
    with st.container(border=True):
        st.markdown(pulse_loading_html(label), unsafe_allow_html=True)
