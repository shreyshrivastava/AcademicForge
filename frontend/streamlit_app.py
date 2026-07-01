import os
import time

import requests
import streamlit as st


BACKEND_URL = os.getenv("ACADEMICFORGE_BACKEND_URL", "http://localhost:8000")


def post_json(path, payload):
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=240)
    response.raise_for_status()
    return response.json()


def get_config():
    response = requests.get(f"{BACKEND_URL}/config", timeout=5)
    response.raise_for_status()
    return response.json()


def search_papers(query):
    return post_json("/search", {"query": query}).get("papers", [])


def summarize_paper(paper):
    return post_json("/summarize", paper).get("summary", "")


def stream_roadmap(papers, summaries):
    response = requests.post(
        f"{BACKEND_URL}/roadmap/stream",
        json={"papers": papers, "summaries": summaries},
        stream=True,
        timeout=240,
    )
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            yield chunk


def roadmap_cache_status(papers, summaries):
    return post_json("/roadmap/cache-status", {"papers": papers, "summaries": summaries})


def paper_label(index, paper):
    rrf_score = float(paper.get("rrf_score", 0.0) or 0.0)
    return f"{index}. {paper['title']} | RRF {rrf_score:.5f}"


def render_results_table(papers):
    st.caption(f"Found {len(papers)} ranked papers.")
    st.dataframe(
        [
            {
                "Rank": index,
                "Title": paper["title"],
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


def render_paper_details(papers, summaries=None):
    summaries = summaries or []
    for index, paper in enumerate(papers):
        summary = summaries[index] if index < len(summaries) else None
        with st.expander(f"{index + 1}. {paper['title']}", expanded=expand_all_results or index == 0):
            st.write(f"**Source:** {paper.get('source', 'arxiv')}")
            st.write(f"**Authors:** {', '.join(paper.get('authors', []))}")
            st.write(f"**Date:** {paper.get('date') or paper.get('published', '')}")
            st.write(f"**URL:** [Open paper]({paper.get('url') or paper.get('link')})")
            snippet = paper["abstract"][:600]
            st.write(f"**Abstract snippet:** {snippet}{'...' if len(paper['abstract']) > 600 else ''}")
            st.write(
                "**Retrieval:** "
                f"BM25 rank `{paper.get('bm25_rank')}`, "
                f"dense rank `{paper.get('dense_rank')}`, "
                f"RRF score `{paper.get('rrf_score', 0.0):.5f}`"
            )
            if debug_mode:
                st.json(paper.get("metadata", {}))
            if summary:
                st.write(f"**Summary:** {summary}")


st.set_page_config(page_title="AcademicForge", page_icon="AF", layout="wide")

st.title("AcademicForge")
st.caption("Find papers, summarize them, and draft an implementation roadmap.")

try:
    config = get_config()
    llm_models = config.get("llm_models", {})
    st.caption(
        "LLM: "
        f"{config['llm_provider']} / "
        f"summary={llm_models.get('summary', 'unknown')} / "
        f"roadmap={llm_models.get('roadmap', 'unknown')}"
    )
except requests.RequestException:
    pass

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

default_query = st.query_params.get("query", "")
debug_mode = st.checkbox("Debug retrieval", value=False)
expand_all_results = st.checkbox("Expand all results", value=True)

research_question = st.text_input(
    "Research question or arXiv link",
    placeholder="Example: https://arxiv.org/abs/1706.03762",
    value=default_query,
)

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
            with st.spinner("Searching arXiv..."):
                st.session_state.papers = search_papers(research_question.strip())
            st.session_state.last_query = research_question.strip()
            st.session_state.summaries = []
            st.session_state.roadmap = ""
            st.session_state.roadmap_elapsed = None
            st.session_state.generated_labels = []
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
    render_results_table(papers)
    render_paper_details(papers)

    labels = [paper_label(index, paper) for index, paper in enumerate(papers, start=1)]
    valid_defaults = [label for label in st.session_state.selected_labels if label in labels]
    selected_labels = st.multiselect(
        "Choose papers for summarization and roadmap",
        labels,
        default=valid_defaults or labels[: min(3, len(labels))],
    )
    st.session_state.selected_labels = selected_labels
    selected_papers = [
        papers[labels.index(label)]
        for label in selected_labels
    ]
    selection_changed_after_generation = (
        bool(st.session_state.generated_labels)
        and selected_labels != st.session_state.generated_labels
    )
    if selection_changed_after_generation:
        st.info("Selection changed. Generate again to update summaries and roadmap for the new papers.")

    if selected_papers:
        st.caption(f"Selected {len(selected_papers)} paper(s) for the roadmap.")
        if len(selected_papers) > 3:
            st.info("For faster and sharper roadmaps, select 2-3 papers unless you want a survey.")
    else:
        st.warning("Select at least one paper to generate a roadmap.")

    if selected_papers and st.session_state.summaries and selected_labels == st.session_state.generated_labels:
        try:
            status = roadmap_cache_status(selected_papers, st.session_state.summaries)
            if status.get("cached"):
                st.success(f"Roadmap cache: ready from {status.get('cache')} cache.")
            else:
                st.info("Roadmap cache: miss. Next generation will run the local MLX model.")
        except requests.RequestException:
            pass

    should_generate = st.button(
        "Generate roadmap from selected papers",
        disabled=not selected_papers,
    )

    if should_generate:
        try:
            with st.spinner("Summarizing selected papers..."):
                st.session_state.summaries = [summarize_paper(paper) for paper in selected_papers]

            with st.spinner("Generating roadmap..."):
                roadmap_started = time.perf_counter()
                roadmap_placeholder = st.empty()
                roadmap_chunks = []
                for chunk in stream_roadmap(selected_papers, st.session_state.summaries):
                    roadmap_chunks.append(chunk)
                    roadmap_placeholder.markdown("".join(roadmap_chunks))
                st.session_state.roadmap = "".join(roadmap_chunks).strip()
                st.session_state.roadmap_elapsed = time.perf_counter() - roadmap_started
                roadmap_placeholder.empty()
            st.session_state.generated_labels = selected_labels
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
    )

    if can_show_generated_output:
        st.subheader("Selected paper summaries")
        render_paper_details(selected_papers, st.session_state.summaries)

    if can_show_generated_output:
        st.subheader("Implementation roadmap")
        if st.session_state.roadmap_elapsed is not None:
            elapsed = st.session_state.roadmap_elapsed
            if elapsed < 1:
                st.success(f"Roadmap loaded from cache in {elapsed:.2f}s.")
            else:
                st.info(f"Roadmap generated locally with MLX in {elapsed:.2f}s.")
        if isinstance(st.session_state.roadmap, str):
            st.markdown(st.session_state.roadmap)
        else:
            st.json(st.session_state.roadmap)

        markdown = "## Research Roadmap\n\n"
        markdown += f"### Research question\n{st.session_state.last_query or research_question.strip()}\n\n"
        for paper, summary in zip(selected_papers, st.session_state.summaries):
            markdown += f"### {paper['title']}\n\n{summary}\n\n"
        markdown += "### Roadmap\n"
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
    st.info("No papers found for this query.")
