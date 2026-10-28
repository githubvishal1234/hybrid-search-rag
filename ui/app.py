"""Streamlit front end. Talks to the FastAPI service over HTTP (API_URL), so UI and API can be
deployed together or apart. With API_URL empty it runs retrieval and generation in-process
instead, which is how it runs on Streamlit Community Cloud (one process, no FastAPI).

uv run streamlit run ui/app.py
"""

import sys
from pathlib import Path

import requests
import streamlit as st

# Streamlit Cloud may install only the dependencies, not this package: import it from src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_rag.config import settings  # noqa: E402

IN_PROCESS = not settings.api_url

MODES = {
    "hybrid_rerank": "Hybrid + rerank: BM25 and vectors fused with RRF, re-scored by a "
    "cross-encoder. Most accurate, slowest.",
    "hybrid": "Hybrid: BM25 and vectors fused with Reciprocal Rank Fusion.",
    "vector": "Vector only: semantic search with bge-small embeddings.",
    "bm25": "BM25 only: keyword search with a finance-aware tokenizer.",
}
EXAMPLES = [
    "What is the TCS rate on remittances under the Liberalised Remittance Scheme for education?",
    "How much will the government borrow relative to the size of the economy next year?",
    "What does section 148A of the Income-tax Act provide?",
    "How should mutual funds handle digital accessibility for investors?",
]

st.set_page_config(page_title="Hybrid Search RAG", page_icon="📑", layout="wide")


@st.cache_data(ttl=30)
def health() -> dict | None:
    if IN_PROCESS:
        from hybrid_rag.api.main import health as local_health

        return local_health()
    try:
        return requests.get(f"{settings.api_url}/health", timeout=5).json()
    except requests.RequestException:
        return None


def ask(question: str, mode: str) -> dict:
    """The /query response, from the API or computed here. Errors stop the page with a message."""
    if IN_PROCESS:
        from hybrid_rag.generation.answer import answer

        try:
            return answer(question, mode)
        except Exception as e:  # LLM quota/outage etc.: show it instead of a stack trace
            st.error(f"{settings.llm_provider} failed: {e}")
            st.stop()
    try:
        r = requests.post(
            f"{settings.api_url}/query", json={"question": question, "mode": mode}, timeout=300
        )
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        st.stop()
    if r.status_code != 200:
        st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    return r.json()


with st.sidebar:
    st.header("Settings")
    mode = st.radio("Retrieval mode", list(MODES), format_func=lambda m: m.replace("_", " + "))
    st.caption(MODES[mode])
    h = health()
    if h:
        st.caption(f"{h['chunks_indexed']:,} chunks indexed · LLM `{h['llm']}`")
    elif not IN_PROCESS:
        st.error(f"API not reachable at {settings.api_url}")
    if not IN_PROCESS:
        st.markdown(f"[API docs]({settings.api_url}/docs)")

st.title("Ask Indian finance regulations")
st.caption(
    "Answers come only from SEBI, Union Budget 2026-27 and IRDAI PDFs, with a citation to the "
    "document and page for every claim. If the documents don't say, it says so."
)

if "question" not in st.session_state:
    st.session_state.question = ""
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES, strict=True):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_area("Question", key="question", height=80)
if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
    with st.spinner("Retrieving and generating (rerank mode and local LLMs are slow)..."):
        res = ask(question, mode)

    if res["refused"]:
        st.info(res["answer"])
    else:
        st.markdown(res["answer"])
    if res["invalid_citations"]:
        dropped = res["invalid_citations"]
        st.warning(f"Dropped citations to sources the model was never given: {dropped}")
    if res["uncited"]:
        st.warning("This answer cites no source. Treat it as unverified.")

    ms = res["latency_ms"]
    a, b, c = st.columns(3)
    a.metric("Retrieval", f"{ms['retrieve']:,} ms")
    b.metric("Generation", f"{ms['generate']:,} ms")
    c.metric("Total", f"{ms['total']:,} ms")

    if res["citations"]:
        st.subheader("Sources")
        for cit in res["citations"]:
            with st.expander(f"{cit['doc']}, {cit['page']}"):
                st.text(cit["snippet"] + " ...")

    with st.expander("Retrieval debug: where each retriever ranked the context chunks"):
        st.caption(
            "vector_rank / bm25_rank: position in each retriever's top 30 (empty = not found by "
            "it). rrf_rank: position after fusion. rerank_score: cross-encoder score."
        )
        st.dataframe(res["retrieval_debug"]["chunks"], use_container_width=True, hide_index=True)
