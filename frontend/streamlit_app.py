import html
import os
import time

import requests
import streamlit as st


BACKEND_URL = os.getenv("ACADEMICFORGE_BACKEND_URL", "http://localhost:8000")
MODE_OPTIONS = {
    "fast": {
        "label": "Fast Mode (Qwen)",
        "short": "Fast",
        "purpose": "Quick insights, shorter responses.",
    },
    "deep": {
        "label": "Deep Mode (Gemma)",
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
CATEGORY_ACCENTS = {
    "Foundational": "#1a73e8",
    "Survey": "#7b1fa2",
    "Recent": "#00897b",
    "Implementation Focused": "#188038",
    "Evaluation Focused": "#f9ab00",
    "Alternative Approach": "#d93025",
    "Contrarian View": "#5f6368",
    "Uncategorized": "#5f6368",
}


def post_json(path, payload):
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=240)
    response.raise_for_status()
    return response.json()


def get_config():
    response = requests.get(f"{BACKEND_URL}/config", timeout=5)
    response.raise_for_status()
    return response.json()


def search_papers(query, categories=None):
    return post_json("/search", {"query": query, "categories": categories or []})


def summarize_paper(paper):
    return post_json("/summarize", paper).get("summary", "")


def generate_paper_roadmap(paper, generation_mode):
    payload = dict(paper)
    payload["generation_mode"] = generation_mode
    return post_json("/roadmap/paper", payload).get("roadmap", "")


def stream_roadmap(papers, summaries, query, generation_mode):
    response = requests.post(
        f"{BACKEND_URL}/roadmap/stream",
        json={
            "papers": papers,
            "summaries": summaries,
            "query": query,
            "generation_mode": generation_mode,
        },
        stream=True,
        timeout=240,
    )
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            yield chunk


def roadmap_cache_status(papers, summaries, query, generation_mode):
    return post_json(
        "/roadmap/cache-status",
        {
            "papers": papers,
            "summaries": summaries,
            "query": query,
            "generation_mode": generation_mode,
        },
    )


def paper_roadmap_cache_status(paper, generation_mode):
    payload = dict(paper)
    payload["generation_mode"] = generation_mode
    return post_json("/roadmap/paper/cache-status", payload)


def paper_cache_id(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper["title"]


def guidance_cache_id(paper, generation_mode):
    return f"{generation_mode}:{paper_cache_id(paper)}"


def paper_label(index, paper):
    rrf_score = float(paper.get("rrf_score", 0.0) or 0.0)
    return f"{index}. {paper['title']} | RRF {rrf_score:.5f}"


def paper_category(paper):
    return paper.get("metadata", {}).get("academicforge_category", "Uncategorized")


def paper_year(paper):
    value = paper.get("date") or paper.get("published", "")
    return str(value)[:4] if value else "unknown"


def applied_focus_categories(research_focus):
    return [category for category in research_focus if category != "Balanced"]


def mode_config(config, generation_mode):
    public_modes = (config or {}).get("generation_modes", {})
    fallback = MODE_OPTIONS[generation_mode]
    return {
        "label": public_modes.get(generation_mode, {}).get("label", fallback["label"]),
        "model": public_modes.get(generation_mode, {}).get("model", "configured model"),
        "purpose": public_modes.get(generation_mode, {}).get("purpose", fallback["purpose"]),
    }


def grouped_by_category(papers):
    groups = {}
    for paper in papers:
        groups.setdefault(paper_category(paper), []).append(paper)
    ordered = [category for category in CATEGORY_OPTIONS if category in groups and category != "Balanced"]
    ordered.extend(sorted(category for category in groups if category not in ordered))
    return [(category, groups[category]) for category in ordered]


def render_category_heading(category, count):
    accent = CATEGORY_ACCENTS.get(category, CATEGORY_ACCENTS["Uncategorized"])
    st.markdown(
        f"""
        <div class="af-section-heading" style="--accent-color: {accent};">
            {html.escape(category)} <span style="font-weight: 400; opacity: 0.72;">({count})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selection_guidance(selected_count, focus_count, generation_mode):
    if selected_count >= 5:
        st.warning(
            "This may take longer. More papers mean more tokens, slower generation, "
            "and less focused recommendations. For sharper synthesis, select 2-4 papers."
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
        .block-container {
            padding-top: 2.2rem;
        }
        .af-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 650;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }
        .af-subtitle {
            text-align: center;
            color: rgba(250, 250, 250, 0.74);
            margin-bottom: 1.2rem;
        }
        .af-search-shell {
            max-width: 920px;
            margin: 0 auto 1.2rem auto;
            padding: 1.15rem 1.15rem 0.8rem 1.15rem;
            border: 1px solid rgba(218, 220, 224, 0.22);
            border-radius: 8px;
            background: rgba(32, 33, 36, 0.58);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.16);
        }
        .af-mode-status {
            max-width: 920px;
            margin: 0 auto 1rem auto;
            border-left: 4px solid #1a73e8;
            padding: 0.62rem 0.8rem;
            border-radius: 6px;
            background: rgba(26, 115, 232, 0.12);
            color: rgba(250, 250, 250, 0.92);
            font-size: 0.92rem;
        }
        .af-section-heading {
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
            padding-left: 0.75rem;
            border-left: 4px solid var(--accent-color);
            font-size: 1.1rem;
            font-weight: 650;
        }
        .paper-card-title {
            color: inherit;
            font-size: 1.05rem;
            font-weight: 650;
            line-height: 1.3;
            margin-bottom: 0.1rem;
        }
        .paper-card-meta {
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.86rem;
            margin-bottom: 0.45rem;
        }
        .paper-card-abstract {
            color: rgba(250, 250, 250, 0.9);
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }
        .paper-chip {
            display: inline-block;
            border: 1px solid #dadce0;
            border-radius: 999px;
            padding: 0.12rem 0.48rem;
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
            color: #3c4043;
            background: #f8fafd;
            font-size: 0.78rem;
            white-space: nowrap;
        }
        .paper-chip-category {
            border-color: rgba(147, 197, 253, 0.45);
            background: rgba(30, 64, 175, 0.45);
            color: #ffffff;
            font-weight: 600;
        }
        .selected-paper-row {
            border-left: 3px solid #1a73e8;
            padding-left: 0.6rem;
            margin-bottom: 0.5rem;
            color: inherit;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_results_table(papers):
    st.caption(f"Found {len(papers)} ranked papers.")
    st.dataframe(
        [
            {
                "Rank": index,
                "Title": paper["title"],
                "Category": paper.get("metadata", {}).get("academicforge_category", "Uncategorized"),
                "Source": paper.get("source", "arxiv"),
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
        st.caption("Retrieval diagnostics and model/runtime details.")
        if config:
            llm_models = config.get("llm_models", {})
            metric_cols = st.columns(3)
            metric_cols[0].metric("Provider", config.get("llm_provider", "unknown"))
            metric_cols[1].metric("Summary model", llm_models.get("summary", "unknown"))
            metric_cols[2].metric("Roadmap model", llm_models.get("roadmap", "unknown"))

        category_counts = {}
        for paper in papers:
            category_counts[paper_category(paper)] = category_counts.get(paper_category(paper), 0) + 1
        st.write("**Evidence category mix**")
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


def render_category_filters(papers):
    categories = sorted({paper_category(paper) for paper in papers})
    if not categories:
        return papers

    selected_categories = st.multiselect(
        "Filter by evidence category",
        categories,
        default=categories,
        help="Narrow the current evidence set by the type of paper you want to work with.",
    )
    if not selected_categories:
        st.warning("Select at least one category to show papers.")
        return []
    return [paper for paper in papers if paper_category(paper) in selected_categories]


def render_selected_evidence(selected_papers):
    with st.container(border=True):
        st.subheader("Selected evidence")
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
        cache_id = paper_cache_id(paper)
        guidance_id = guidance_cache_id(paper, generation_mode)
        label = paper_label(all_labels.index(paper) + 1, paper) if paper in all_labels else paper_label(index + 1, paper)
        title = html.escape(paper["title"])
        authors = html.escape(", ".join(paper.get("authors", [])[:5]) or "Unknown authors")
        if len(paper.get("authors", [])) > 5:
            authors += ", et al."
        url = paper.get("url") or paper.get("link")
        source = html.escape(paper.get("source", "arxiv"))
        category = html.escape(paper_category(paper))
        snippet = html.escape(paper["abstract"][:650] + ("..." if len(paper["abstract"]) > 650 else ""))

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="paper-card-title">{title}</div>
                <div class="paper-card-meta">{authors} · {html.escape(paper_year(paper))} · {source}</div>
                <div class="paper-card-abstract">{snippet}</div>
                <span class="paper-chip paper-chip-category">{category}</span>
                """,
                unsafe_allow_html=True,
            )

            action_cols = st.columns([1.25, 1, 1, 4])
            if selectable:
                checked = action_cols[0].checkbox(
                    "Select",
                    value=label in selected_set,
                    key=f"select-paper-{scope}-{cache_id}",
                )
                if checked:
                    selected_set.add(label)
                else:
                    selected_set.discard(label)
            if url:
                action_cols[1].link_button("Open", url)
            if show_guidance_controls:
                if action_cols[2].button("Guidance", key=f"paper-roadmap-button-{scope}-{cache_id}"):
                    try:
                        status = paper_roadmap_cache_status(paper, generation_mode)
                        if status.get("cached"):
                            st.session_state.paper_roadmaps[guidance_id] = generate_paper_roadmap(paper, generation_mode)
                        else:
                            with st.spinner("Generating paper guidance..."):
                                st.session_state.paper_roadmaps[guidance_id] = generate_paper_roadmap(paper, generation_mode)
                    except requests.ConnectionError:
                        st.error("The backend is not running. Start it with: uvicorn backend.app:app --reload")
                    except requests.HTTPError as exc:
                        st.error(f"The backend returned an error: {exc.response.text}")
                    except requests.RequestException as exc:
                        st.error(f"Could not reach the backend: {exc}")

            if summary:
                st.markdown("**Summary**")
                st.markdown(summary)

            if guidance_id in st.session_state.paper_roadmaps:
                st.markdown("**Paper guidance**")
                st.markdown(st.session_state.paper_roadmaps[guidance_id])

    if selectable:
        valid_all_labels = [paper_label(index, paper) for index, paper in enumerate(all_labels, start=1)]
        st.session_state.selected_labels = [
            label for label in valid_all_labels if label in selected_set
        ]


def render_paper_details(papers, summaries=None, show_roadmap_controls=False, scope="results", generation_mode="fast"):
    summaries = summaries or []
    for index, paper in enumerate(papers):
        summary = summaries[index] if index < len(summaries) else None
        cache_id = paper_cache_id(paper)
        guidance_id = guidance_cache_id(paper, generation_mode)
        with st.expander(f"{index + 1}. {paper['title']}", expanded=index == 0):
            st.write(f"**Source:** {paper.get('source', 'arxiv')}")
            st.write(f"**Authors:** {', '.join(paper.get('authors', []))}")
            st.write(f"**Date:** {paper.get('date') or paper.get('published', '')}")
            st.write(
                "**Category:** "
                f"{paper_category(paper)}"
            )
            st.write(f"**URL:** [Open paper]({paper.get('url') or paper.get('link')})")
            snippet = paper["abstract"][:600]
            st.write(f"**Abstract snippet:** {snippet}{'...' if len(paper['abstract']) > 600 else ''}")
            if summary:
                st.write(f"**Summary:** {summary}")
            if show_roadmap_controls:
                if st.button("Guidance", key=f"paper-roadmap-button-{scope}-{cache_id}"):
                    try:
                        status = paper_roadmap_cache_status(paper, generation_mode)
                        if status.get("cached"):
                            st.session_state.paper_roadmaps[guidance_id] = generate_paper_roadmap(paper, generation_mode)
                        else:
                            with st.spinner("Generating paper guidance..."):
                                st.session_state.paper_roadmaps[guidance_id] = generate_paper_roadmap(paper, generation_mode)
                    except requests.ConnectionError:
                        st.error("The backend is not running. Start it with: uvicorn backend.app:app --reload")
                    except requests.HTTPError as exc:
                        st.error(f"The backend returned an error: {exc.response.text}")
                    except requests.RequestException as exc:
                        st.error(f"Could not reach the backend: {exc}")
                if guidance_id in st.session_state.paper_roadmaps:
                    st.markdown("**Paper guidance**")
                    st.markdown(st.session_state.paper_roadmaps[guidance_id])


st.set_page_config(page_title="AcademicForge", page_icon="AF", layout="wide")
apply_card_styles()

st.markdown('<div class="af-title">AcademicForge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="af-subtitle">Question -> Research -> Decision -> Prototype</div>',
    unsafe_allow_html=True,
)

try:
    config = get_config()
    config = get_config()
except requests.RequestException:
    config = None

if "papers" not in st.session_state:
    st.session_state.papers = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = []
if "summaries" not in st.session_state:
    st.session_state.summaries = []
if "roadmap" not in st.session_state:
    st.session_state.roadmap = ""
if "roadmap_elapsed" not in st.session_state:
    st.session_state.roadmap_elapsed = None
if "generated_labels" not in st.session_state:
    st.session_state.generated_labels = []
if "generated_mode" not in st.session_state:
    st.session_state.generated_mode = ""
if "paper_roadmaps" not in st.session_state:
    st.session_state.paper_roadmaps = {}
if "generation_mode" not in st.session_state:
    st.session_state.generation_mode = "fast"
if "research_focus" not in st.session_state:
    st.session_state.research_focus = ["Balanced"]
if "last_search_focus" not in st.session_state:
    st.session_state.last_search_focus = []
if "search_message" not in st.session_state:
    st.session_state.search_message = ""

default_query = st.query_params.get("query", "")

st.markdown('<div class="af-search-shell">', unsafe_allow_html=True)
research_question = st.text_input(
    "Research question or arXiv link",
    placeholder="Ask a research question or paste an arXiv link",
    value=default_query,
)
mode_labels = [MODE_OPTIONS["fast"]["label"], MODE_OPTIONS["deep"]["label"]]
current_mode_index = 0 if st.session_state.generation_mode == "fast" else 1
selected_mode_label = st.radio(
    "Analysis mode",
    mode_labels,
    index=current_mode_index,
    horizontal=True,
    help=(
        "Fast Mode: Quick insights, shorter responses. "
        "Deep Mode: Detailed analysis, prototype guidance."
    ),
)
st.session_state.generation_mode = "deep" if selected_mode_label == MODE_OPTIONS["deep"]["label"] else "fast"
research_focus = st.multiselect(
    "Research Focus",
    CATEGORY_OPTIONS,
    default=st.session_state.research_focus,
    help=(
        "Choose the paper types you want more answers from. "
        "Balanced keeps the default evidence mix. One or two focused categories is usually best."
    ),
)
if not research_focus:
    research_focus = ["Balanced"]
if "Balanced" in research_focus and len(research_focus) > 1:
    research_focus = [category for category in research_focus if category != "Balanced"]
st.session_state.research_focus = research_focus
focus_categories = applied_focus_categories(research_focus)
st.markdown("</div>", unsafe_allow_html=True)

active_mode = mode_config(config, st.session_state.generation_mode)
st.markdown(
    f"""
    <div class="af-mode-status">
        <strong>{html.escape(active_mode["label"])}</strong>
        &nbsp;using <code>{html.escape(active_mode["model"])}</code><br/>
        {html.escape(active_mode["purpose"])}
    </div>
    """,
    unsafe_allow_html=True,
)

render_selection_guidance(0, len(focus_categories), st.session_state.generation_mode)

auto_run_search = (
    st.query_params.get("run") == "1"
    and research_question.strip()
    and st.session_state.last_query != research_question.strip()
)
should_search = st.button("Search papers", type="primary") or auto_run_search

if should_search:
    if not research_question.strip():
        st.warning("Please enter a research question.")
    else:
        try:
            with st.spinner("Searching live research sources..."):
                search_payload = search_papers(research_question.strip(), focus_categories)
                st.session_state.papers = search_payload.get("papers", [])
                st.session_state.search_message = search_payload.get("message", "")
            st.session_state.last_query = research_question.strip()
            st.session_state.last_search_focus = list(focus_categories)
            st.session_state.summaries = []
            st.session_state.roadmap = ""
            st.session_state.roadmap_elapsed = None
            st.session_state.generated_labels = []
            st.session_state.generated_mode = ""
            st.session_state.paper_roadmaps = {}
            labels = [
                paper_label(index, paper)
                for index, paper in enumerate(st.session_state.papers, start=1)
            ]
            st.session_state.selected_labels = labels[: min(3, len(labels))]
        except requests.ConnectionError:
            st.error("The backend is not running. Start it with: uvicorn backend.app:app --reload")
        except requests.HTTPError as exc:
            st.error(f"The backend returned an error: {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")

papers = st.session_state.papers
if papers:
    st.subheader("Search results")
    labels = [paper_label(index, paper) for index, paper in enumerate(papers, start=1)]

    if st.session_state.last_search_focus:
        st.caption(
            "Research focus applied on search: "
            + ", ".join(st.session_state.last_search_focus)
        )
    else:
        st.caption("Research focus applied on search: Balanced")

    filtered_papers = render_category_filters(papers)
    st.caption(f"Showing {len(filtered_papers)} of {len(papers)} selected evidence papers.")

    action_cols = st.columns([1, 1, 4])
    filtered_labels = [
        paper_label(papers.index(paper) + 1, paper)
        for paper in filtered_papers
    ]
    if action_cols[0].button("Select visible"):
        merged = set(st.session_state.selected_labels)
        merged.update(filtered_labels)
        st.session_state.selected_labels = [label for label in labels if label in merged]
    if action_cols[1].button("Clear selection"):
        st.session_state.selected_labels = []

    for category, category_papers in grouped_by_category(filtered_papers):
        render_category_heading(category, len(category_papers))
        render_paper_cards(
            category_papers,
            papers,
            show_guidance_controls=True,
            selectable=True,
            scope=f"results-{category.lower().replace(' ', '-')}",
            generation_mode=st.session_state.generation_mode,
        )
    render_technical_dashboard(papers, config)

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
        st.info("Selection or mode changed. Generate again to update synthesis.")

    if selected_papers:
        st.caption(f"Selected {len(selected_papers)} paper(s) for research synthesis.")
        render_selected_evidence(selected_papers)
        render_selection_guidance(
            len(selected_papers),
            len(st.session_state.last_search_focus),
            st.session_state.generation_mode,
        )
    else:
        st.warning("Select at least one paper to generate a synthesis.")

    if selected_papers and st.session_state.summaries and selected_labels == st.session_state.generated_labels:
        try:
            status = roadmap_cache_status(
                selected_papers,
                st.session_state.summaries,
                st.session_state.last_query or research_question.strip(),
                st.session_state.generation_mode,
            )
            if status.get("cached"):
                st.success(f"Synthesis cache: ready from {status.get('cache')} cache.")
            else:
                st.info(f"Synthesis cache: miss. Next generation will run {active_mode['label']}.")
        except requests.RequestException:
            pass

    should_generate = st.button(
        "Generate research synthesis",
        disabled=not selected_papers,
    )

    if should_generate:
        try:
            with st.spinner("Summarizing selected papers..."):
                st.session_state.summaries = [summarize_paper(paper) for paper in selected_papers]

            with st.spinner("Synthesizing research evidence..."):
                roadmap_started = time.perf_counter()
                roadmap_placeholder = st.empty()
                roadmap_chunks = []
                for chunk in stream_roadmap(
                    selected_papers,
                    st.session_state.summaries,
                    st.session_state.last_query or research_question.strip(),
                    st.session_state.generation_mode,
                ):
                    roadmap_chunks.append(chunk)
                    roadmap_placeholder.markdown("".join(roadmap_chunks))
                st.session_state.roadmap = "".join(roadmap_chunks).strip()
                st.session_state.roadmap_elapsed = time.perf_counter() - roadmap_started
                roadmap_placeholder.empty()
            st.session_state.generated_labels = selected_labels
            st.session_state.generated_mode = st.session_state.generation_mode
        except requests.ConnectionError:
            st.error("The backend is not running. Start it with: uvicorn backend.app:app --reload")
        except requests.HTTPError as exc:
            st.error(f"The backend returned an error: {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")

    can_show_generated_output = (
        st.session_state.summaries
        and st.session_state.roadmap
        and selected_labels == st.session_state.generated_labels
        and st.session_state.generation_mode == st.session_state.generated_mode
    )

    if can_show_generated_output:
        st.subheader("Selected paper summaries")
        render_paper_cards(
            selected_papers,
            selected_papers,
            summaries=st.session_state.summaries,
            show_guidance_controls=False,
            selectable=False,
            scope="selected",
            generation_mode=st.session_state.generation_mode,
        )

    if can_show_generated_output:
        st.subheader("Research synthesis")
        if st.session_state.roadmap_elapsed is not None:
            elapsed = st.session_state.roadmap_elapsed
            if elapsed < 1:
                st.success(f"Synthesis loaded from cache in {elapsed:.2f}s.")
            else:
                st.info(f"Synthesis generated with {active_mode['label']} in {elapsed:.2f}s.")
        if isinstance(st.session_state.roadmap, str):
            st.markdown(st.session_state.roadmap)
        else:
            st.json(st.session_state.roadmap)

        markdown = "## Research Synthesis\n\n"
        markdown += f"### Research question\n{st.session_state.last_query or research_question.strip()}\n\n"
        for paper, summary in zip(selected_papers, st.session_state.summaries):
            markdown += f"### {paper['title']}\n\n{summary}\n\n"
        markdown += "### Synthesis\n"
        markdown += (
            st.session_state.roadmap
            if isinstance(st.session_state.roadmap, str)
            else str(st.session_state.roadmap)
        )
        st.download_button(
            "Download Markdown",
            markdown,
            file_name="academicforge-roadmap.md",
            mime="text/markdown",
        )
elif st.session_state.last_query:
    st.info(st.session_state.search_message or "No papers found for this query.")
