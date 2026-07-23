import html
import os
import re
import time
import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from frontend.api_client import APIClient

api_client = APIClient()


def display_model_name(model):
    if not model:
        return "configured model"
    model_name = str(model).split("/")[-1]
    return model_name.removesuffix("-4bit")


def display_mode_label(mode, model):
    mode_name = "Fast Mode" if mode == "fast" else "Deep Mode"
    return f"{mode_name} - {display_model_name(model)}"


MODE_OPTIONS = {
    "fast": {
        "label": "Fast Mode - gemma-2-2b-it",
        "short": "Fast",
        "purpose": "Quick insights, shorter responses.",
    },
    "deep": {
        "label": "Deep Mode - deepseek-v4-pro",
        "short": "Deep",
        "purpose": "Detailed analysis, prototype guidance.",
    },
}
CATEGORY_OPTIONS = [
    "Balanced",
    "Foundational",
    "Survey",
    "Recent",
    "Implementation Focused",
    "Evaluation Focused",
    "Alternative Approach",
    "Contrarian View",
]
RESEARCH_LENS_OPTIONS = [
    "Balanced",
    "Foundational",
    "Survey",
    "Implementation Focused",
    "Evaluation Focused",
    "Alternative Approach",
    "Contrarian View",
]
RESEARCH_LENS_DESCRIPTIONS = {
    "Balanced": "General-purpose mix of evidence.",
    "Foundational": "Prioritize core and highly cited work.",
    "Survey": "Prioritize review and survey papers.",
    "Implementation Focused": "Prioritize practical methods and systems.",
    "Evaluation Focused": "Prioritize benchmarks, metrics, and comparisons.",
    "Alternative Approach": "Prioritize different solution paths.",
    "Contrarian View": "Prioritize papers that challenge common assumptions.",
}
CATEGORY_ACCENTS = {
    "Foundational": "#5B8DBE",
    "Survey": "#9C7ECF",
    "Recent": "#4AA99C",
    "Implementation Focused": "#6FA85C",
    "Evaluation Focused": "#D4A24C",
    "Alternative Approach": "#7C8FA6",
    "Contrarian View": "#8A8F98",
    "Uncategorized": "#6B7280",
}


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


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def accent_rgba(hex_color, alpha):
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


def post_json(path, payload):
    return api_client.post_json(path, payload)


def get_config():
    return api_client.get_config()


def search_papers(query, categories=None):
    return post_json("/search", {"query": query, "categories": categories or []})


def summarize_paper(paper, generation_mode="fast"):
    payload = dict(paper)
    payload["generation_mode"] = generation_mode
    return post_json("/summarize", payload).get("summary", "")


def generate_paper_guidance(paper, generation_mode):
    payload = dict(paper)
    payload["generation_mode"] = generation_mode
    response = post_json("/paper-guidance", payload)
    return response.get("guidance", "")


def stream_research_plan(papers, summaries, query, generation_mode):
    yield from api_client.stream_post(
        "/research-plan/stream",
        {
            "papers": papers,
            "summaries": summaries,
            "query": query,
            "generation_mode": generation_mode,
        }
    )


def paper_widget_id(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper["title"]


def paper_display_id(paper, generation_mode):
    return f"{generation_mode}:{paper_widget_id(paper)}"


def paper_label(index, paper):
    rrf_score = float(paper.get("rrf_score", 0.0) or 0.0)
    return f"{index}. {paper['title']} | RRF {rrf_score:.5f}"


def paper_category(paper):
    return paper.get("metadata", {}).get("academicforge_category", "Uncategorized")


def paper_year(paper):
    value = paper.get("date") or paper.get("published", "")
    return str(value)[:4] if value else "unknown"


def citation_label(paper):
    metadata = paper.get("metadata", {}) or {}
    citation_count = metadata.get("citation_count")
    influential_count = metadata.get("influential_citation_count")
    if citation_count is None:
        return ""
    label = f"{citation_count:,} citations"
    if influential_count:
        label += f" · {influential_count:,} influential"
    return label


def applied_focus_categories(research_lens):
    return [] if research_lens == "Balanced" else [research_lens]


def mode_config(config, generation_mode):
    public_modes = (config or {}).get("generation_modes", {})
    fallback = MODE_OPTIONS[generation_mode]
    model = public_modes.get(generation_mode, {}).get("model", "configured model")
    return {
        "label": display_mode_label(generation_mode, model) if model != "configured model" else fallback["label"],
        "model": model,
        "purpose": public_modes.get(generation_mode, {}).get("purpose", fallback["purpose"]),
    }


def render_selection_guidance(selected_count, focus_count, generation_mode):
    if selected_count >= 5:
        st.warning(
            "This may take longer. More papers mean more tokens, slower generation, "
            "and less focused recommendations. For a sharper Research Plan, select 2-4 papers."
        )
    elif generation_mode == "deep" and selected_count >= 4:
        st.warning(
            "Deep Mode is more detailed and slower. For the best Gemma output, use 2-3 papers."
        )
    elif selected_count == 4:
        st.info("Four papers is workable, but 2-3 papers usually gives sharper recommendations.")

    if focus_count >= 4:
        st.warning(
            "Broad research focus selected. Choosing many paper types can make results less focused."
        )
    elif focus_count == 3:
        st.info("Three focus categories is broad. For focused retrieval, 1-2 categories is usually better.")


def apply_card_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap');

        :root {
            --af-ember: #ED1C24;
            --af-ember-hover: #C4151B;
            --af-ember-soft: rgba(237, 28, 36, 0.14);
            --af-hairline: rgba(255, 255, 255, 0.10);
            --af-hairline-strong: rgba(255, 255, 255, 0.20);
            --af-surface: rgba(255, 255, 255, 0.035);
            --af-surface-raised: rgba(255, 255, 255, 0.055);
            --af-text-hi: rgba(240, 240, 238, 0.96);
            --af-text-mid: rgba(240, 240, 238, 0.70);
            --af-text-low: rgba(240, 240, 238, 0.48);
            --af-font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            font-family: var(--af-font);
        }

        .block-container {
            max-width: 1600px;
            padding-top: clamp(1.1rem, 2.4vw, 2.4rem);
            padding-left: clamp(0.75rem, 2.4vw, 2.5rem);
            padding-right: clamp(0.75rem, 2.4vw, 2.5rem);
        }

        .af-title {
            text-align: center;
            font-family: var(--af-font);
            font-size: clamp(1.75rem, 4.2vw, 3rem);
            font-weight: 700;
            letter-spacing: 0;
            color: var(--af-text-hi);
            margin-bottom: 0.3rem;
        }
        .af-title::after {
            content: '';
            display: block;
            width: 46px;
            height: 3px;
            margin: 0.55rem auto 0 auto;
            background: var(--af-ember);
            border-radius: 2px;
        }
        .af-subtitle {
            text-align: center;
            color: var(--af-text-mid);
            font-size: clamp(0.82rem, 1.5vw, 1.05rem);
            letter-spacing: 0;
            margin-bottom: clamp(1rem, 2vw, 1.6rem);
        }
        .af-helper-card {
            max-width: 1600px;
            margin: 0 auto 1rem auto;
            padding: 0.95rem 1.1rem;
            border: 1px solid var(--af-hairline);
            border-radius: 10px;
            background: var(--af-surface);
        }
        .af-helper-title {
            font-weight: 600;
            color: var(--af-text-hi);
            margin-bottom: 0.3rem;
        }
        .af-helper-text {
            color: var(--af-text-mid);
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .af-empty-state {
            max-width: 1600px;
            margin: 0.35rem auto 1rem auto;
            padding: 0.9rem 1rem;
            border-radius: 8px;
            border: 1px solid var(--af-hairline);
            background: var(--af-surface);
            color: var(--af-text-mid);
        }
        .st-key-search_shell {
            max-width: 1600px;
            margin: 0 auto 1.2rem auto;
        }
        .st-key-search_shell [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-search_shell[data-testid="stVerticalBlockBorderWrapper"] {
            padding: clamp(0.95rem, 1.8vw, 1.45rem);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.045);
        }
        .af-control-caption {
            color: var(--af-text-mid);
            font-size: 0.82rem;
            margin: 0.15rem 0 0.55rem 0;
        }
        div[data-testid="stTextInput"] input {
            min-height: 46px;
            border-radius: 8px;
            border: 1px solid var(--af-hairline-strong);
            background: rgba(255,255,255,0.06);
            color: var(--af-text-hi);
            font-size: 0.92rem;
            font-weight: 500;
        }
        div[data-testid="stTextInput"] label,
        div[data-testid="stPills"] label {
            color: var(--af-text-mid);
            font-size: 0.96rem;
            font-weight: 700;
        }
        div[data-testid="stPills"] {
            margin-top: 0.75rem;
        }
        div[data-testid="stPills"] button {
            min-height: 36px;
            border-radius: 9px;
            border: 1px solid var(--af-hairline-strong);
            background: rgba(255,255,255,0.035);
            color: var(--af-text-mid);
            font-size: 0.84rem;
            font-weight: 700;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        div[data-testid="stPills"] button[aria-pressed="true"] {
            border-color: rgba(237, 28, 36, 0.65);
            background: rgba(237, 28, 36, 0.16);
            color: #ff5a60;
        }
        div[data-testid="stPills"] button:hover {
            border-color: rgba(237, 28, 36, 0.50);
            color: var(--af-text-hi);
        }
        .af-lens-heading {
            color: var(--af-text-hi);
            font-size: 0.96rem;
            font-weight: 700;
            margin-top: 0.9rem;
            margin-bottom: 0.15rem;
        }
        .af-lens-subtitle {
            color: var(--af-text-mid);
            font-size: 0.84rem;
            line-height: 1.4;
            margin-bottom: 0.55rem;
        }
        .af-lens-description {
            color: var(--af-text-mid);
            font-size: 0.84rem;
            line-height: 1.45;
            margin-top: 0.45rem;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        div[data-testid="stRadio"] label {
            min-height: 36px;
            border: 1px solid var(--af-hairline-strong);
            border-radius: 9px;
            padding: 7px 14px;
            background: rgba(255,255,255,0.035);
            cursor: pointer;
        }
        div[data-testid="stRadio"] label > div:first-child {
            display: none;
        }
        div[data-testid="stRadio"] label p {
            color: var(--af-text-mid);
            font-size: 0.84rem;
            font-weight: 700;
        }
        div[data-testid="stRadio"] label:has(input:checked) {
            background: rgba(237, 28, 36, 0.22);
            border-color: var(--af-ember);
        }
        div[data-testid="stRadio"] label:has(input:checked) p {
            color: #ff6a70;
            font-weight: 700;
        }
        @media (max-width: 640px) {
            div[data-testid="stRadio"] label {
                padding: 6px 11px;
            }
            div[data-testid="stRadio"] label p {
                font-size: 0.78rem;
            }
        }
        div.stButton > button[kind="primary"] {
            min-height: 46px;
            border-radius: 8px;
            background: var(--af-ember);
            border-color: var(--af-ember);
            color: #FFFFFF;
            font-weight: 700;
            font-size: 0.92rem;
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--af-ember-hover);
            border-color: var(--af-ember-hover);
            color: #FFFFFF;
        }
        .af-section-heading {
            display: flex;
            align-items: center;
            gap: 9px;
            margin-top: 1.7rem;
            margin-bottom: 0.7rem;
            font-family: var(--af-font);
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--af-text-hi);
        }
        .af-section-marker {
            width: 10px;
            height: 10px;
            border-radius: 3px;
            background: var(--accent-color);
            flex-shrink: 0;
        }
        .af-section-count {
            font-weight: 400;
            color: var(--af-text-low);
            font-size: 0.95rem;
        }
        .paper-card-title {
            color: var(--af-text-hi);
            font-family: var(--af-font);
            font-size: 1.12rem;
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 0.15rem;
        }
        .paper-card-meta {
            color: var(--af-text-low);
            font-family: var(--af-font);
            font-size: 0.78rem;
            letter-spacing: 0;
            margin-bottom: 0.55rem;
        }
        .paper-card-abstract {
            color: var(--af-text-mid);
            font-size: 0.93rem;
            line-height: 1.5;
            margin-bottom: 0.6rem;
            text-align: justify;
        }
        .paper-chip {
            display: inline-block;
            border: 1px solid var(--af-hairline-strong);
            border-radius: 999px;
            padding: 0.14rem 0.55rem;
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
            color: var(--af-text-mid);
            background: var(--af-surface);
            font-size: 0.76rem;
            white-space: nowrap;
        }
        .paper-chip-category {
            font-weight: 600;
        }
        .af-pulse-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            padding: 1.4rem 0;
        }
        .af-pulse-dots {
            display: flex;
            gap: 8px;
        }
        .af-pulse-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--af-ember);
            animation: af-dot-breathe 1.2s ease-in-out infinite;
        }
        .af-pulse-dot:nth-child(2) { animation-delay: 0.15s; }
        .af-pulse-dot:nth-child(3) { animation-delay: 0.3s; }
        @keyframes af-dot-breathe {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
            40% { transform: scale(1); opacity: 1; }
        }
        .af-pulse-label {
            font-size: 0.8rem;
            color: var(--af-text-mid);
        }
        @media (prefers-reduced-motion: reduce) {
            .af-pulse-dot { animation: none; opacity: 0.8; }
        }
        .af-insight-block {
            border-left: 2px solid var(--af-ember);
            border-radius: 0 8px 8px 0;
            background: rgba(255,255,255,0.025);
            border-top: 1px solid var(--af-hairline);
            border-right: 1px solid var(--af-hairline);
            border-bottom: 1px solid var(--af-hairline);
            padding-left: 0.7rem;
            padding-top: 0.65rem;
            padding-right: 0.8rem;
            padding-bottom: 0.7rem;
            margin-top: 0.75rem;
            margin-bottom: 0.2rem;
        }
        .af-insight-label {
            font-family: var(--af-font);
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0;
            color: var(--af-ember);
            margin-bottom: 3px;
            font-weight: 700;
        }
        .af-insight-text {
            font-size: 0.87rem;
            color: var(--af-text-mid);
            line-height: 1.55;
            white-space: pre-line;
        }
        .selected-paper-row {
            border-left: 2px solid var(--af-ember);
            padding-left: 0.65rem;
            margin-bottom: 0.5rem;
            color: var(--af-text-hi);
            font-size: 0.9rem;
        }
        .research-plan-control-title {
            color: var(--af-text-hi);
            font-weight: 600;
            margin-bottom: 0.15rem;
        }
        .research-plan-control-meta {
            color: var(--af-text-mid);
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .research-plan-stage-empty {
            border: 1px solid var(--af-hairline);
            border-radius: 10px;
            background: rgba(255,255,255,0.025);
            padding: 1.45rem;
            text-align: center;
            margin-top: 0.8rem;
        }
        .research-plan-stage-title {
            color: var(--af-text-hi);
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .research-plan-stage-copy {
            color: var(--af-text-low);
            font-size: 0.86rem;
        }
        .syn-heading {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0.8rem;
            font-family: var(--af-font);
            font-size: 1rem;
            font-weight: 700;
            color: var(--af-text-hi);
        }
        .syn-marker {
            width: 6px;
            height: 6px;
            border-radius: 2px;
            background: var(--af-ember);
            flex-shrink: 0;
        }
        .syn-heading-label {
            text-transform: capitalize;
        }
        .syn-field-label {
            font-family: var(--af-font);
            font-weight: 700;
            color: var(--af-text-hi);
            font-size: 0.92rem;
            margin-top: 0.6rem;
            margin-bottom: 0.35rem;
        }
        .syn-field-value {
            color: var(--af-text-mid);
            font-size: 0.90rem;
            line-height: 1.55;
            margin-bottom: 0.5rem;
        }
        .syn-list {
            margin-left: 1.2rem;
            color: var(--af-text-mid);
            font-size: 0.90rem;
            line-height: 1.6;
        }
        .syn-list li {
            margin-bottom: 0.3rem;
        }
        .syn-badge {
            display: inline-block;
            background: var(--af-surface-raised);
            border: 1px solid var(--af-hairline);
            border-radius: 4px;
            padding: 0.25rem 0.6rem;
            margin-right: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.80rem;
            color: var(--af-text-mid);
        }
        .syn-body {
            color: var(--af-text-mid);
            font-size: 0.90rem;
            line-height: 1.6;
        }
        .cite {
            color: var(--af-ember);
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(config, generation_mode):
    if config is None:
        st.markdown(
            """
            <div class="af-helper-card">
                <div class="af-helper-title">Backend unavailable</div>
                <div class="af-helper-text">The interface is ready, but the backend is not responding yet. Start it with <strong>./start.sh</strong> and refresh the page.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if config.get("llm_provider") == "cloud_demo":
        st.markdown(
            """
            <div class="af-helper-card">
                <div class="af-helper-title">Hosted demo version</div>
                <div class="af-helper-text">
                    This public demo keeps the original AcademicForge interface, but uses a
                    cloud-safe deterministic runtime. It does not run the local FastAPI,
                    MLX/ROCm, full Gemma, or paid Fireworks path by default. Research Plan
                    generation is limited to one use per IP address, and live search may
                    fall back to sample data if public academic APIs rate-limit.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return


def render_results_table(papers):
    st.caption(f"Found {len(papers)} ranked papers.")
    st.dataframe(
        [
            {
                "Rank": index,
                "Title": paper["title"],
                "Category": paper.get("metadata", {}).get("academicforge_category", "Uncategorized"),
                "Source": paper.get("source", "arxiv"),
                "Citations": paper.get("metadata", {}).get("citation_count"),
                "Influential": paper.get("metadata", {}).get("influential_citation_count"),
                "BM25": paper.get("bm25_rank"),
                "Dense": paper.get("dense_rank"),
                "RRF": round(float(paper.get("rrf_score", 0.0) or 0.0), 5),
                "URL": paper.get("url") or paper.get("link"),
            }
            for index, paper in enumerate(papers, start=1)
        ],
        hide_index=True,
        width="stretch",
    )


def render_technical_dashboard(papers, config=None):
    with st.expander("Technical dashboard", expanded=False):
        st.caption("Retrieval diagnostics and model details.")
        if config:
            llm_models = config.get("llm_models", {})
            metric_cols = st.columns(3)
            metric_cols[0].metric("Summary model", display_model_name(llm_models.get("summary")))
            metric_cols[1].metric("Guidance model", display_model_name(llm_models.get("guidance")))
            metric_cols[2].metric(
                "Research Plan model",
                display_model_name(llm_models.get("research_plan")),
            )

        category_counts = {}
        for paper in papers:
            category_counts[paper_category(paper)] = category_counts.get(paper_category(paper), 0) + 1
        st.write("**Paper category mix**")
        st.dataframe(
            [
                {"Category": category, "Count": count}
                for category, count in sorted(category_counts.items())
            ],
            hide_index=True,
            width="stretch",
        )

        st.write("**Retrieval scores**")
        render_results_table(papers)

        with st.expander("Raw metadata", expanded=False):
            for index, paper in enumerate(papers, start=1):
                st.write(f"**{index}. {paper['title']}**")
                st.json(paper.get("metadata", {}))


def render_selected_evidence(selected_papers):
    with st.container(border=True):
        st.subheader("Selected papers")
        if not selected_papers:
            st.caption("Choose papers from the result cards.")
            return
        for paper in selected_papers:
            st.markdown(
                f"""
                <div class="selected-paper-row">
                  <strong>{html.escape(paper["title"])}</strong><br/>
                  {html.escape(paper_category(paper))} · {html.escape(paper_year(paper))}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_paper_cards(
    papers,
    all_labels,
    summaries=None,
    show_guidance_controls=False,
    selectable=False,
    scope="results",
    generation_mode="fast",
):
    summaries = summaries or []
    selected_set = set(st.session_state.selected_labels)

    for index, paper in enumerate(papers):
        summary = summaries[index] if index < len(summaries) else None
        paper_widget_key = paper_widget_id(paper)
        display_key = paper_display_id(paper, generation_mode)
        paper_outputs = st.session_state.paper_outputs.setdefault(display_key, {})
        label = paper_label(all_labels.index(paper) + 1, paper) if paper in all_labels else paper_label(index + 1, paper)
        title = html.escape(paper["title"])
        authors = html.escape(", ".join(paper.get("authors", [])[:5]) or "Unknown authors")
        if len(paper.get("authors", [])) > 5:
            authors += ", et al."
        source = html.escape(paper.get("source", "arxiv"))
        category_name = paper_category(paper)
        category = html.escape(category_name)
        category_accent = CATEGORY_ACCENTS.get(category_name, CATEGORY_ACCENTS["Uncategorized"])
        abstract = html.escape(paper.get("abstract", ""))
        citation_text = citation_label(paper)
        meta_parts = [
            authors,
            html.escape(paper_year(paper)),
            source,
        ]
        if citation_text:
            meta_parts.append(html.escape(citation_text))
        metadata_line = " · ".join(meta_parts)

        inline_summary = summary or paper_outputs.get("summary")
        inline_guidance = paper_outputs.get("guidance")

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="paper-card-title">{title}</div>
                <div class="paper-card-meta">{metadata_line}</div>
                <span class="paper-chip paper-chip-category" style="border-color: {accent_rgba(category_accent, 0.45)}; background: {accent_rgba(category_accent, 0.16)}; color: {category_accent};">{category}</span>
                """,
                unsafe_allow_html=True,
            )
            # Render abstract natively as markdown so LaTeX math equations are processed
            # The newlines ensure Streamlit parses the markdown inside the HTML block
            st.markdown(f"<div class='paper-card-abstract'>\n\n{paper.get('abstract', '')}\n\n</div>", unsafe_allow_html=True)
            if selectable:
                action_cols = st.columns([1.1, 1.05, 1.05, 3.8])
                select_col, summary_col, guidance_col = (
                    action_cols[0],
                    action_cols[1],
                    action_cols[2],
                )
            else:
                action_cols = st.columns([1.05, 1.05, 4.0])
                select_col, summary_col, guidance_col = (
                    None,
                    action_cols[0],
                    action_cols[1],
                )

            if selectable:
                checked = select_col.checkbox(
                    "Select",
                    value=label in selected_set,
                    key=f"select-paper-{scope}-{paper_widget_key}",
                )
                if checked:
                    selected_set.add(label)
                else:
                    selected_set.discard(label)
            summary_clicked = summary_col.button("Summary", key=f"paper-summary-button-{scope}-{paper_widget_key}")
            guidance_clicked = guidance_col.button("Guidance", key=f"paper-guidance-button-{scope}-{paper_widget_key}")

            panel_placeholder = st.empty()
            if summary_clicked:
                panel_placeholder.markdown(
                    paper_insight_loading_html("Summary", "Summarizing this paper..."),
                    unsafe_allow_html=True,
                )
                try:
                    inline_summary = summarize_paper(paper, st.session_state.generation_mode)
                    paper_outputs["summary"] = inline_summary
                    panel_placeholder.empty()
                except requests.ConnectionError:
                    panel_placeholder.error(
                        "The backend is not running. Start it with: ./start.sh"
                    )
                except requests.HTTPError as exc:
                    panel_placeholder.error(f"The backend returned an error: {exc.response.text}")
                except requests.RequestException as exc:
                    panel_placeholder.error(f"Could not reach the backend: {exc}")

            if guidance_clicked:
                panel_placeholder.markdown(
                    paper_insight_loading_html("Guidance", "Generating guidance..."),
                    unsafe_allow_html=True,
                )
                try:
                    inline_guidance = generate_paper_guidance(paper, generation_mode)
                    paper_outputs["guidance"] = inline_guidance
                    panel_placeholder.empty()
                except requests.ConnectionError:
                    panel_placeholder.error(
                        "The backend is not running. Start it with: ./start.sh"
                    )
                except requests.HTTPError as exc:
                    panel_placeholder.error(f"The backend returned an error: {exc.response.text}")
                except requests.RequestException as exc:
                    panel_placeholder.error(f"Could not reach the backend: {exc}")

            if inline_summary:
                st.markdown(paper_insight_panel_html("Summary", inline_summary), unsafe_allow_html=True)
            if inline_guidance:
                st.markdown(paper_insight_panel_html("Guidance", inline_guidance), unsafe_allow_html=True)

    if selectable:
        valid_all_labels = [paper_label(index, paper) for index, paper in enumerate(all_labels, start=1)]
        st.session_state.selected_labels = [
            label for label in valid_all_labels if label in selected_set
        ]


st.set_page_config(page_title="AcademicForge", page_icon="AF", layout="wide")
apply_card_styles()

st.markdown('<div class="af-title">AcademicForge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="af-subtitle">Question → Papers → Plan</div>',
    unsafe_allow_html=True,
)

try:
    config = get_config()
except requests.RequestException:
    config = None

render_status_banner(config, st.session_state.generation_mode if "generation_mode" in st.session_state else "fast")

if "papers" not in st.session_state:
    st.session_state.papers = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = []
if "summaries" not in st.session_state:
    st.session_state.summaries = []
if "research_plan" not in st.session_state:
    st.session_state.research_plan = ""
if "research_plan_elapsed" not in st.session_state:
    st.session_state.research_plan_elapsed = None
if "generated_labels" not in st.session_state:
    st.session_state.generated_labels = []
if "generated_mode" not in st.session_state:
    st.session_state.generated_mode = ""
if "paper_outputs" not in st.session_state:
    st.session_state.paper_outputs = {}
if "generation_mode" not in st.session_state:
    st.session_state.generation_mode = "fast"
if "research_lens" not in st.session_state:
    previous_focus = st.session_state.get("research_focus", ["Balanced"])
    if isinstance(previous_focus, list) and previous_focus:
        st.session_state.research_lens = (
            previous_focus[0] if previous_focus[0] in RESEARCH_LENS_OPTIONS else "Balanced"
        )
    else:
        st.session_state.research_lens = "Balanced"
if "last_search_focus" not in st.session_state:
    st.session_state.last_search_focus = []
if "search_message" not in st.session_state:
    st.session_state.search_message = ""

default_query = st.query_params.get("query", "")

with st.container(border=True, key="search_shell"):
    st.session_state.generation_mode = st.radio(
        "Analysis Mode",
        options=["fast", "deep"],
        format_func=lambda x: mode_config(config, x)["label"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    search_cols = st.columns([5, 1.05], vertical_alignment="bottom", gap="medium")
    research_question = search_cols[0].text_input(
        "Research question",
        placeholder="Ask a research question",
        value=default_query,
        label_visibility="collapsed",
    )
    auto_run_search = (
        st.query_params.get("run") == "1"
        and research_question.strip()
        and st.session_state.last_query != research_question.strip()
    )
    manual_search = search_cols[1].button("Search", type="primary", width="stretch")

    st.markdown(
        """
        <div class="af-lens-heading">Research Lens</div>
        <div class="af-lens-subtitle">Choose how AcademicForge explores and synthesizes research.</div>
        """,
        unsafe_allow_html=True,
    )
    research_lens = st.radio(
        "Research Lens",
        RESEARCH_LENS_OPTIONS,
        index=RESEARCH_LENS_OPTIONS.index(st.session_state.research_lens),
        horizontal=True,
        label_visibility="collapsed",
        key="research_lens",
    )
    st.markdown(
        f'<div class="af-lens-description">{html.escape(RESEARCH_LENS_DESCRIPTIONS[research_lens])}</div>',
        unsafe_allow_html=True,
    )
    


focus_categories = applied_focus_categories(research_lens)

active_mode = mode_config(config, st.session_state.generation_mode)

render_selection_guidance(0, len(focus_categories), st.session_state.generation_mode)

should_search = manual_search or auto_run_search

if should_search:
    if not research_question.strip():
        st.warning("Please enter a research question.")
    else:
        try:
            search_placeholder = st.empty()
            search_placeholder.markdown(
                pulse_loading_html("Searching live research sources..."),
                unsafe_allow_html=True,
            )
            search_payload = search_papers(research_question.strip(), focus_categories)
            st.session_state.papers = search_payload.get("papers", [])
            st.session_state.search_message = search_payload.get("message", "")
            search_placeholder.empty()
            st.session_state.last_query = research_question.strip()
            st.session_state.last_search_focus = list(focus_categories)
            st.session_state.summaries = []
            st.session_state.research_plan = ""
            st.session_state.research_plan_elapsed = None
            st.session_state.generated_labels = []
            st.session_state.generated_mode = ""
            st.session_state.paper_outputs = {}
            labels = [
                paper_label(index, paper)
                for index, paper in enumerate(st.session_state.papers, start=1)
            ]
            st.session_state.selected_labels = labels[: min(3, len(labels))]
        except requests.ConnectionError:
            search_placeholder.empty()
            st.error("The backend is not running. Start it with: ./start.sh")
        except requests.HTTPError as exc:
            search_placeholder.empty()
            st.error(f"The backend returned an error: {exc.response.text}")
        except requests.RequestException as exc:
            search_placeholder.empty()
            st.error(f"Could not reach the backend: {exc}")

papers = st.session_state.papers
if papers:
    st.subheader("Paper Results")
    labels = [paper_label(index, paper) for index, paper in enumerate(papers, start=1)]

    if focus_categories:
        filtered_papers = [
            p for p in papers
            if p.get("metadata", {}).get("academicforge_category") in focus_categories
        ]
    else:
        filtered_papers = papers
    st.caption(f"{len(papers)} papers retrieved. Select papers to generate a Research Plan.")

    filtered_labels = [
        paper_label(papers.index(paper) + 1, paper)
        for paper in filtered_papers
    ]

    render_paper_cards(
        filtered_papers,
        papers,
        show_guidance_controls=True,
        selectable=True,
        scope="results",
        generation_mode=st.session_state.generation_mode,
    )

    selected_labels = [label for label in st.session_state.selected_labels if label in labels]
    st.session_state.selected_labels = selected_labels
    selected_papers = [
        papers[labels.index(label)]
        for label in selected_labels
    ]
    selection_changed_after_generation = (
        bool(st.session_state.generated_labels)
        and (
            selected_labels != st.session_state.generated_labels
            or st.session_state.generation_mode != st.session_state.generated_mode
        )
    )
    if selection_changed_after_generation:
        st.info("Selection or mode changed. Generate again to update the Research Plan.")

    if selected_papers:
        st.caption(f"Selected {len(selected_papers)} paper(s) for the combined Research Plan.")
        render_selected_evidence(selected_papers)
        render_selection_guidance(
            len(selected_papers),
            len(st.session_state.last_search_focus),
            st.session_state.generation_mode,
        )
    else:
        st.warning("Select at least one paper to generate a Research Plan.")

    with st.container(border=True):
        control_cols = st.columns([3, 1.25])
        control_cols[0].markdown(
            f"""
            <div class="research-plan-control-title">Research Plan</div>
            <div class="research-plan-control-meta">
                {len(selected_papers)} selected paper(s) · Active mode: {html.escape(active_mode['label'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
        should_generate = control_cols[1].button(
            "Generate Research Plan",
            disabled=not selected_papers,
            type="primary",
            width="stretch",
        )

    if should_generate:
        plan_placeholder = st.empty()
        try:
            selected_summaries = []
            for paper in selected_papers:
                with plan_placeholder.container():
                    render_research_plan_loading_state("Summarizing selected evidence...")
                summary = summarize_paper(paper, st.session_state.generation_mode)
                selected_summaries.append(summary)
            st.session_state.summaries = selected_summaries

            research_plan_started = time.perf_counter()
            with plan_placeholder.container():
                render_research_plan_loading_state("Streaming Research Plan...")
            research_plan_chunks = []
            for chunk in stream_research_plan(
                selected_papers,
                st.session_state.summaries,
                st.session_state.last_query or research_question.strip(),
                st.session_state.generation_mode,
            ):
                research_plan_chunks.append(chunk)
            st.session_state.research_plan = strip_evidence_citations("".join(research_plan_chunks)).strip()
            st.session_state.research_plan_elapsed = time.perf_counter() - research_plan_started
            plan_placeholder.empty()
            st.session_state.generated_labels = selected_labels
            st.session_state.generated_mode = st.session_state.generation_mode
        except requests.ConnectionError:
            plan_placeholder.empty()
            st.error("The backend is not running. Start it with: ./start.sh")
        except requests.HTTPError as exc:
            plan_placeholder.empty()
            st.error(f"The backend returned an error: {exc.response.text}")
        except requests.RequestException as exc:
            plan_placeholder.empty()
            st.error(f"Could not reach the backend: {exc}")

    can_show_generated_output = (
        st.session_state.summaries
        and st.session_state.research_plan
        and selected_labels == st.session_state.generated_labels
        and st.session_state.generation_mode == st.session_state.generated_mode
    )

    if not can_show_generated_output and not should_generate:
        render_research_plan_empty_state()

    if can_show_generated_output:
        if st.session_state.research_plan_elapsed is not None:
            elapsed = st.session_state.research_plan_elapsed
            st.info(f"Research Plan generated with {active_mode['label']} in {elapsed:.2f}s.")
        render_research_plan_panel(st.session_state.research_plan)

        markdown = "## Research Plan\n\n"
        markdown += f"### Research question\n{st.session_state.last_query or research_question.strip()}\n\n"
        for paper, summary in zip(selected_papers, st.session_state.summaries):
            markdown += f"### {paper['title']}\n\n{summary}\n\n"
        markdown += "### Combined Research Plan\n"
        markdown += (
            st.session_state.research_plan
            if isinstance(st.session_state.research_plan, str)
            else str(st.session_state.research_plan)
        )
        st.download_button(
            "Download Markdown",
            markdown,
            file_name="academicforge-research-plan.md",
            mime="text/markdown",
        )
elif st.session_state.last_query:
    st.markdown(
        '<div class="af-empty-state">No evidence was found for that query. Try broadening the wording, switching research focus, or using a more specific question.</div>',
        unsafe_allow_html=True,
    )
