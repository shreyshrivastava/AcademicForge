import html
import time
import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from frontend.api_client import create_api_client
from frontend.workflow import paper_output_key
from backend.search_context import FIELD_PRESETS, contextualize_query
from frontend.styles import apply_card_styles
from frontend.components import (
    paper_insight_loading_html,
    paper_insight_panel_html,
    pulse_loading_html,
    render_research_plan_empty_state,
    render_research_plan_loading_state,
    render_research_plan_panel,
    strip_evidence_citations,
)

api_client = create_api_client()


def display_model_name(model):
    if not model:
        return "configured model"
    model_name = str(model).split("/")[-1]
    return model_name.removesuffix("-4bit")


def display_mode_label(mode, model):
    mode_name = "Fast" if mode == "fast" else "Deep"
    return f"{mode_name} · {display_model_name(model)}"


MODE_OPTIONS = {
    "fast": {
        "label": "Fast · gemma-2-2b-it",
        "short": "Fast",
        "purpose": "Shorter answers, quicker.",
    },
    "deep": {
        "label": "Deep · deepseek-v4-pro",
        "short": "Deep",
        "purpose": "Longer analysis and build guidance.",
    },
}
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
    "Balanced": "A mix of paper types.",
    "Foundational": "Core, highly cited work first.",
    "Survey": "Reviews and surveys first.",
    "Implementation Focused": "Methods and systems you can build first.",
    "Evaluation Focused": "Benchmarks, metrics and comparisons first.",
    "Alternative Approach": "Different ways of solving the problem first.",
    "Contrarian View": "Papers that question the usual assumptions first.",
}
def post_json(path, payload):
    response = api_client.post_json(path, payload)
    if response.get("generation", {}).get("fallback"):
        st.warning(response["generation"].get("fallback_message") or "The cloud model didn't respond, so this answer comes from the local model.")
    return response


def get_config():
    # Reused for 30s per session: every click reruns the script, and config rarely changes.
    cached = st.session_state.get("_config_cache")
    if cached and time.monotonic() - cached[0] < 30:
        return cached[1]
    value = api_client.get_config()
    st.session_state._config_cache = (time.monotonic(), value)
    return value


def search_papers(query, categories=None, domain=""):
    return post_json("/search", {"query": query, "categories": categories or [], "domain": domain})


def summarize_paper(paper, generation_mode="fast"):
    payload = dict(paper)
    payload["generation_mode"] = generation_mode
    if (config or {}).get("standalone"):
        payload["query"] = st.session_state.last_query
    key = paper_display_id(paper, generation_mode)
    outputs = st.session_state.paper_outputs.setdefault(key, {})
    retry_preview = (config or {}).get("standalone") and outputs.get("summary_generation", {}).get("fallback")
    if outputs.get("summary") and not retry_preview:
        if outputs.get("summary_generation", {}).get("fallback"):
            st.caption(outputs["summary_generation"].get("fallback_message") or "This saved summary came from the local fallback model.")
        return outputs["summary"]
    response = post_json("/summarize", payload)
    result = response.get("summary", "")
    outputs["summary"] = result
    outputs["summary_generation"] = response.get("generation", {})
    return result


def generate_paper_guidance(paper, generation_mode):
    payload = dict(paper)
    payload["generation_mode"] = generation_mode
    if (config or {}).get("standalone"):
        payload["query"] = st.session_state.last_query
    response = post_json("/paper-guidance", payload)
    return response.get("guidance", "")


def stream_research_plan(papers, summaries, query, generation_mode, domain=""):
    yield from api_client.stream_post(
        "/research-plan/stream",
        {
            "papers": papers,
            "summaries": summaries,
            "query": query,
            "domain": domain,
            "generation_mode": generation_mode,
        }
    )


def paper_widget_id(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper["title"]


def paper_display_id(paper, generation_mode):
    return paper_output_key(paper, generation_mode, (config or {}).get("llm_models", {}))


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
        "label": (public_modes.get(generation_mode, {}).get("label") if (config or {}).get("standalone") else None)
        or (display_mode_label(generation_mode, model) if model != "configured model" else fallback["label"]),
        "model": model,
        "purpose": public_modes.get(generation_mode, {}).get("purpose", fallback["purpose"]),
    }


def selection_hint(selected_count, generation_mode):
    if selected_count == 0:
        return "Pick at least one paper."
    if selected_count >= 5:
        return "Five or more papers makes the plan slower and less focused. Two to four works best."
    if generation_mode == "deep" and selected_count >= 4:
        return "Deep mode gets slow with four or more papers. Two or three works best."
    if selected_count == 4:
        return "Four works. Two or three usually gives a sharper plan."
    return ""


def render_status_banner(config):
    if config is None:
        st.error("Can't reach the backend. Start it with ./start.sh, then refresh this page.")
        return

    if config.get("standalone"):
        st.info("This is the lightweight demo. It searches live sources and ranks papers with BM25. "
                "Summaries and plans come from a small CPU model plus extracted evidence, and no API keys are needed.")
        from frontend.small_model import get_model
        model_state = config.get("model_state", "disabled")
        if model_state == "loading":
            st.caption("Downloading FLAN-T5-small (about 97 MB, first run only). Search and extracted evidence work in the meantime.")
            @st.fragment(run_every="3s")
            def check_model():
                if get_model().status() != "loading":
                    st.rerun()
            check_model()
        elif model_state == "ready":
            st.caption("FLAN-T5-small is running on CPU. Check its drafts against the papers.")
        elif model_state == "unavailable":
            st.warning("The small model didn't load, so answers use extracted evidence only.")
            if st.button("Retry small model"):
                get_model().retry()
                st.rerun()
        else:
            st.caption("The small model is off, so answers use extracted evidence only.")


def ranking_note(config):
    if not config or config.get("standalone"):
        return "ranked with BM25"
    mode = config.get("retrieval_mode", "score_fusion")
    return "ranked with BM25 + BGE score fusion" if mode == "score_fusion" else f"ranked with {mode}"


def section_heading(title, meta=""):
    st.markdown(
        f'<div class="af-section"><div class="af-section-title">{html.escape(title)}</div><span>{html.escape(meta)}</span></div>',
        unsafe_allow_html=True,
    )


def show_request_error(target, exc):
    if isinstance(exc, requests.ConnectionError):
        target.error("Can't reach the backend. Start it with ./start.sh and try again.")
    elif isinstance(exc, requests.HTTPError):
        target.error(f"The request failed: {exc.response.text}")
    else:
        target.error(f"The backend didn't answer: {exc}")


def render_paper_cards(papers, all_papers, generation_mode="fast"):
    selected_set = set(st.session_state.selected_labels)

    for index, paper in enumerate(papers):
        paper_widget_key = paper_widget_id(paper)
        paper_outputs = st.session_state.paper_outputs.setdefault(paper_display_id(paper, generation_mode), {})
        label = paper_label(all_papers.index(paper) + 1, paper) if paper in all_papers else paper_label(index + 1, paper)
        authors = ", ".join(paper.get("authors", [])[:4]) or "Unknown authors"
        if len(paper.get("authors", [])) > 4:
            authors += " et al."
        meta_parts = [html.escape(authors), html.escape(paper_year(paper)), html.escape(paper.get("source", "arxiv"))]
        if citation_label(paper):
            meta_parts.append(html.escape(citation_label(paper)))
        url = paper.get("url") or paper.get("link") or ""
        title = html.escape(paper["title"])
        title_html = f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{title}</a>' if url else title
        select_key = f"select-paper-results-{paper_widget_key}"

        with st.container(key=f"card-{index}"):
            st.markdown(
                f'<div class="af-paper-title">{title_html}</div>'
                f'<div class="af-meta"><span class="af-cat">{html.escape(paper_category(paper))}</span> · {" · ".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Abstract"):
                # Markdown so LaTeX in abstracts renders.
                st.markdown(paper.get("abstract", ""))

            select_col, summary_col, guidance_col, _ = st.columns([1.3, 0.9, 0.9, 3], vertical_alignment="center")
            checked = select_col.checkbox("Use in plan", value=label in selected_set, key=select_key)
            if checked:
                selected_set.add(label)
            else:
                selected_set.discard(label)
            summary_clicked = summary_col.button("Summary", type="tertiary", key=f"paper-summary-button-results-{paper_widget_key}")
            guidance_clicked = guidance_col.button("Guidance", type="tertiary", key=f"paper-guidance-button-results-{paper_widget_key}")

            panel = st.empty()
            if summary_clicked:
                panel.markdown(paper_insight_loading_html("Summary", "Reading the abstract"), unsafe_allow_html=True)
                try:
                    paper_outputs["summary"] = summarize_paper(paper, generation_mode)
                    panel.empty()
                except requests.RequestException as exc:
                    show_request_error(panel, exc)
            if guidance_clicked:
                panel.markdown(paper_insight_loading_html("Guidance", "Working out how to use it"), unsafe_allow_html=True)
                try:
                    paper_outputs["guidance"] = generate_paper_guidance(paper, generation_mode)
                    panel.empty()
                except requests.RequestException as exc:
                    show_request_error(panel, exc)

            if paper_outputs.get("summary"):
                st.markdown(paper_insight_panel_html("Summary", paper_outputs["summary"]), unsafe_allow_html=True)
            if paper_outputs.get("guidance"):
                st.markdown(paper_insight_panel_html("Guidance", paper_outputs["guidance"]), unsafe_allow_html=True)

    valid_labels = [paper_label(index, paper) for index, paper in enumerate(all_papers, start=1)]
    st.session_state.selected_labels = [label for label in valid_labels if label in selected_set]


def plan_markdown(selected_papers, research_question):
    markdown = "## Research plan\n\n"
    if (config or {}).get("standalone"):
        markdown += "Draft from abstracts only, not checked against full papers. Ranking: BM25.\n\n"
    markdown += f"### Research question\n{st.session_state.last_query or research_question}\n\n"
    markdown += f"Search field: {st.session_state.last_search_domain or 'All fields'}\n\n"
    for paper, summary in zip(selected_papers, st.session_state.summaries):
        markdown += f"### {paper['title']}\n\n{summary}\n\n"
    plan = st.session_state.research_plan
    return markdown + "### Combined plan\n" + (plan if isinstance(plan, str) else str(plan))


@st.fragment
def render_results(research_question):
    # A fragment: ticking a paper or asking for a summary reruns only this part, not the search form.
    papers = st.session_state.papers
    labels = [paper_label(index, paper) for index, paper in enumerate(papers, start=1)]
    section_heading("Papers", f"{len(papers)} found · {ranking_note(config)}")

    focus = st.session_state.last_search_focus
    shown = [p for p in papers if p.get("metadata", {}).get("academicforge_category") in focus] if focus else papers
    render_paper_cards(shown, papers, generation_mode=st.session_state.generation_mode)

    selected_labels = [label for label in st.session_state.selected_labels if label in labels]
    st.session_state.selected_labels = selected_labels
    selected_papers = [papers[labels.index(label)] for label in selected_labels]
    active_mode = mode_config(config, st.session_state.generation_mode)

    section_heading("Research plan")
    with st.container(key="plan_bar"):
        info_col, button_col = st.columns([3, 1.2], vertical_alignment="center")
        count = len(selected_papers)
        hint = selection_hint(count, st.session_state.generation_mode)
        stale = bool(st.session_state.generated_labels) and (
            selected_labels != st.session_state.generated_labels
            or st.session_state.generation_mode != st.session_state.generated_mode
        )
        if stale:
            hint = "You changed the papers or the model. Write the plan again to update it."
        info_col.markdown(
            f'<div class="af-plan-meta"><strong>{count} {"paper" if count == 1 else "papers"}</strong> · '
            f'{html.escape(active_mode["label"])}<br>{html.escape(hint)}</div>',
            unsafe_allow_html=True,
        )
        should_generate = button_col.button("Write plan", disabled=not selected_papers, type="primary", width="stretch")

    if should_generate:
        plan_placeholder = st.empty()
        try:
            st.session_state.research_plan = ""
            st.session_state.generated_labels = []
            summaries = []
            for position, paper in enumerate(selected_papers, start=1):
                with plan_placeholder.container():
                    render_research_plan_loading_state(f"Summarizing paper {position} of {count}")
                summaries.append(summarize_paper(paper, st.session_state.generation_mode))
            st.session_state.summaries = summaries

            started = time.perf_counter()
            with plan_placeholder.container():
                render_research_plan_loading_state("Writing the plan")
            chunks = []
            for chunk in stream_research_plan(
                selected_papers,
                st.session_state.summaries,
                st.session_state.last_query or research_question,
                st.session_state.generation_mode,
                st.session_state.last_search_domain,
            ):
                chunks.append(chunk)
                plan_placeholder.markdown("".join(chunks))
            if api_client.last_generation.get("fallback"):
                st.warning(api_client.last_generation.get("fallback_message") or "The cloud model didn't respond, so this plan comes from the local model.")
            st.session_state.research_plan = strip_evidence_citations("".join(chunks)).strip()
            st.session_state.research_plan_elapsed = time.perf_counter() - started
            plan_placeholder.empty()
            st.session_state.generated_labels = selected_labels
            st.session_state.generated_mode = st.session_state.generation_mode
        except requests.RequestException as exc:
            show_request_error(plan_placeholder, exc)

    ready = (
        st.session_state.summaries
        and st.session_state.research_plan
        and selected_labels == st.session_state.generated_labels
        and st.session_state.generation_mode == st.session_state.generated_mode
    )
    if not ready and not should_generate:
        render_research_plan_empty_state()
    if ready:
        render_research_plan_panel(st.session_state.research_plan)
        meta_col, download_col = st.columns([3, 1.2], vertical_alignment="center")
        if st.session_state.research_plan_elapsed is not None:
            meta_col.caption(f"Written by {active_mode['label']} in {st.session_state.research_plan_elapsed:.1f}s.")
        download_col.download_button(
            "Download Markdown",
            plan_markdown(selected_papers, research_question),
            file_name="academicforge-research-plan.md",
            mime="text/markdown",
            width="stretch",
        )


st.set_page_config(page_title="AcademicForge", page_icon=":material/article:", layout="centered")
apply_card_styles()

st.markdown(
    '<div class="af-brand">Academic<span>Forge</span></div>'
    '<div class="af-tagline">Ask a research question, pick the papers that fit, and get a build plan that cites them.</div>',
    unsafe_allow_html=True,
)

try:
    config = get_config()
except requests.RequestException:
    config = None

render_status_banner(config)

output_revision = (config or {}).get("output_revision", "model-v1")
if st.session_state.get("output_revision") != output_revision:
    for name in ("paper_outputs", "summaries", "research_plan", "generated_labels", "generated_mode"):
        st.session_state.pop(name, None)
    st.session_state.output_revision = output_revision

defaults = {
    "papers": [], "last_query": "", "selected_labels": [], "summaries": [], "research_plan": "",
    "research_plan_elapsed": None, "generated_labels": [], "generated_mode": "", "paper_outputs": {},
    "generation_mode": "fast", "last_search_focus": [], "search_message": "", "last_search_domain": "",
    "last_search_query": "", "last_search_identity": None,
}
for name, value in defaults.items():
    st.session_state.setdefault(name, value)
if "research_lens" not in st.session_state:
    previous_focus = st.session_state.get("research_focus", ["Balanced"])
    first = previous_focus[0] if isinstance(previous_focus, list) and previous_focus else "Balanced"
    st.session_state.research_lens = first if first in RESEARCH_LENS_OPTIONS else "Balanced"

search_cols = st.columns([5, 1], vertical_alignment="bottom")
research_question = search_cols[0].text_input(
    "Research question",
    placeholder="For example: reducing hallucinations in retrieval augmented generation",
    value=st.query_params.get("query", ""),
    label_visibility="collapsed",
).strip()
manual_search = search_cols[1].button("Search", type="primary", width="stretch")

option_cols = st.columns([1.2, 1.2, 1])
field_choice = option_cols[0].selectbox(
    "Field", ["All fields", *FIELD_PRESETS, "Other field…"], key="search_field",
    help="The subject you mean, so RAG in AI and ragas in music don't mix. A custom field matches its words in titles, abstracts and categories.",
)
research_lens = option_cols[1].selectbox(
    "Lens", RESEARCH_LENS_OPTIONS, key="research_lens",
    help=" ".join(f"{name}: {text}" for name, text in RESEARCH_LENS_DESCRIPTIONS.items()),
)
st.session_state.generation_mode = option_cols[2].radio(
    "Model",
    options=["fast"] if (config or {}).get("standalone") else ["fast", "deep"],
    format_func=lambda x: mode_config(config, x)["label"].split(" · ")[0],
    horizontal=True,
    help=" ".join(f"{mode_config(config, m)['label']}: {mode_config(config, m)['purpose']}" for m in ("fast", "deep")),
)
custom_field = ""
if field_choice == "Other field…":
    custom_field = st.text_input("Your field", placeholder="For example: quantum computing", max_chars=120, key="custom_search_field")
search_domain = custom_field.strip() if field_choice == "Other field…" else ("" if field_choice == "All fields" else field_choice)
search_is_valid = field_choice != "Other field…" or bool(custom_field.strip())
interpreted_query = contextualize_query(research_question, search_domain) if research_question else ""
if search_domain and interpreted_query != research_question:
    st.caption(f"Searching for: {interpreted_query}")

focus_categories = applied_focus_categories(research_lens)
search_identity = (research_question, search_domain, tuple(focus_categories))
auto_run_search = (
    st.query_params.get("run") == "1" and bool(research_question)
    and st.session_state.last_search_identity != search_identity
)

if manual_search or auto_run_search:
    if not research_question:
        st.warning("Type a research question first.")
    elif not search_is_valid:
        st.warning("Enter a field, or switch back to All fields.")
    else:
        search_placeholder = st.empty()
        search_placeholder.markdown(pulse_loading_html("Searching arXiv and Semantic Scholar"), unsafe_allow_html=True)
        try:
            search_payload = search_papers(research_question, focus_categories, search_domain)
            search_placeholder.empty()
            for warning in search_payload.get("warnings", []):
                st.warning(warning)
            st.session_state.papers = search_payload.get("papers", [])
            st.session_state.search_message = search_payload.get("message", "")
            st.session_state.last_query = research_question
            st.session_state.last_search_domain = search_domain
            st.session_state.last_search_query = search_payload.get("search_query") or interpreted_query
            st.session_state.last_search_identity = search_identity
            st.session_state.last_search_focus = list(focus_categories)
            st.session_state.summaries = []
            st.session_state.research_plan = ""
            st.session_state.research_plan_elapsed = None
            st.session_state.generated_labels = []
            st.session_state.generated_mode = ""
            st.session_state.paper_outputs = {}
            labels = [paper_label(index, paper) for index, paper in enumerate(st.session_state.papers, start=1)]
            st.session_state.selected_labels = labels[:3]
        except requests.RequestException as exc:
            show_request_error(search_placeholder, exc)

if st.session_state.last_query and st.session_state.last_search_identity != search_identity:
    st.info("The question or filters changed. Press Search to update the papers.")

if st.session_state.papers:
    render_results(research_question)
elif st.session_state.last_query:
    st.markdown(
        f'<div class="af-empty">{html.escape(st.session_state.search_message or "No papers matched. Try a broader question or a different field.")}</div>',
        unsafe_allow_html=True,
    )
