# frontend/streamlit_app.py
import requests
import streamlit as st


BACKEND_URL = "http://localhost:8000"


def post_json(path, payload):
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=240)
    response.raise_for_status()
    return response.json()


def search_papers(query):
    return post_json("/search", {"query": query}).get("papers", [])


def summarize_paper(paper):
    return post_json("/summarize", paper).get("summary", "")


def generate_roadmap(papers, summaries):
    return post_json("/roadmap", {"papers": papers, "summaries": summaries}).get("roadmap", "")


st.set_page_config(page_title="AcademicForge", page_icon="AF", layout="wide")

st.title("AcademicForge")
st.caption("Find papers, summarize them, and draft an implementation roadmap.")

default_query = st.query_params.get("query", "")

research_question = st.text_input(
    "Research question or arXiv link",
    placeholder="Example: https://arxiv.org/abs/1706.03762",
    value=default_query,
)

should_generate = st.button("Generate roadmap", type="primary") or st.query_params.get("run") == "1"

if should_generate:
    if not research_question.strip():
        st.warning("Please enter a research question.")
    else:
        try:
            with st.spinner("Searching arXiv..."):
                papers = search_papers(research_question.strip())

            with st.spinner("Summarizing papers..."):
                summaries = [summarize_paper(paper) for paper in papers]

            with st.spinner("Generating roadmap..."):
                roadmap = generate_roadmap(papers, summaries)

            st.subheader("Search results")
            if not papers:
                st.info("No papers found for this query.")
            for index, (paper, summary) in enumerate(zip(papers, summaries)):
                with st.expander(paper["title"], expanded=index == 0):
                    st.write(f"**Authors:** {', '.join(paper['authors'])}")
                    st.write(f"**Date:** {paper['date']}")
                    st.write(f"**Link:** [Open paper]({paper['link']})")
                    st.write(paper["abstract"])
                    st.write(f"**Summary:** {summary}")

            st.subheader("Implementation roadmap")
            if isinstance(roadmap, str):
                st.markdown(roadmap)
            else:
                st.json(roadmap)

            markdown = "## Research Roadmap\n\n"
            markdown += f"### Research question\n{research_question.strip()}\n\n"
            for paper, summary in zip(papers, summaries):
                markdown += f"### {paper['title']}\n\n{summary}\n\n"
            markdown += "### Roadmap\n"
            markdown += roadmap if isinstance(roadmap, str) else str(roadmap)
            st.download_button(
                "Download Markdown",
                markdown,
                file_name="academicforge-roadmap.md",
                mime="text/markdown",
            )
        except requests.ConnectionError:
            st.error("The backend is not running. Start it with: uvicorn backend.app:app --reload")
        except requests.HTTPError as exc:
            st.error(f"The backend returned an error: {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
