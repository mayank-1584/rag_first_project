"""
Streamlit front-end for the RAG assistant — styled version.

Run with:
    streamlit run application.py
"""
import streamlit as st

from src.search import RAGSearch

st.set_page_config(
    page_title="Research Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.4rem 1rem 1.8rem 1rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-family: 'Sora', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #8A8FA3;
        font-size: 0.98rem;
        margin: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12131A;
        border-right: 1px solid #23242E;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: 'Sora', sans-serif;
    }

    .stat-card {
        background: linear-gradient(135deg, #1B1D2A 0%, #23253A 100%);
        border: 1px solid #2E3046;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .stat-card .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #8A8FA3;
    }
    .stat-card .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #E9E9F2;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.4rem;
    }

    /* Source cards */
    .source-card {
        background-color: #1A1B24;
        border: 1px solid #2A2C3A;
        border-left: 3px solid #8B5CF6;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.55rem;
    }
    .source-card .source-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }
    .source-card .source-tag {
        font-size: 0.72rem;
        font-weight: 600;
        color: #B497F5;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .source-card .source-score {
        font-size: 0.72rem;
        color: #6E7183;
        background: #23242E;
        padding: 0.1rem 0.5rem;
        border-radius: 20px;
    }
    .source-card .source-text {
        font-size: 0.86rem;
        color: #C4C6D4;
        line-height: 1.45;
    }

    /* Example question chips */
    .chip-btn button {
        border-radius: 20px !important;
        border: 1px solid #33354A !important;
        background: #1A1B24 !important;
        color: #C4C6D4 !important;
        font-size: 0.85rem !important;
        padding: 0.35rem 0.9rem !important;
    }
    .chip-btn button:hover {
        border-color: #8B5CF6 !important;
        color: #E9E9F2 !important;
    }
</style>
"""

EXAMPLE_QUESTIONS = [
    "What is BERT?",
    "How does self-attention work?",
    "What problem do residual connections solve?",
]


@st.cache_resource(show_spinner="Loading knowledge base and models... (first load can take a minute)")
def load_rag() -> RAGSearch:
    return RAGSearch()


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📄  {len(sources)} source chunk{'s' if len(sources) != 1 else ''} used"):
        for i, src in enumerate(sources, start=1):
            preview = src["text"][:500].strip()
            more = "…" if len(src["text"]) > 500 else ""
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-head">
                        <span class="source-tag">Chunk {i}</span>
                        <span class="source-score">distance {src['distance']:.3f}</span>
                    </div>
                    <div class="source-text">{preview}{more}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_query(rag: RAGSearch, query: str, top_k: int, show_sources: bool) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Searching and generating answer..."):
            try:
                answer, sources = rag.search_with_sources(query, top_k=top_k)
            except Exception as e:  # noqa: BLE001
                answer, sources = f"Something went wrong while generating the answer: {e}", []
        st.markdown(answer)
        if show_sources and sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>📚 Research Assistant</h1>
            <p>Ask questions about the indexed knowledge base — grounded answers, cited sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=3)
        show_sources = st.checkbox("Show retrieved sources", value=True)
        st.divider()

        try:
            rag = load_rag()
            n_chunks = len(rag.vectorstore.metadata)
        except Exception:
            rag = None
            n_chunks = "—"

        st.markdown("### 📊 Knowledge Base")
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Indexed chunks</div>
                <div class="stat-value">{n_chunks}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Embedding model</div>
                <div class="stat-value" style="font-size:0.95rem;">all-MiniLM-L6-v2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if rag is None:
        st.error("Failed to initialize the RAG system.")
        st.info(
            "Check that `GROQ_API_KEY` is set (see `.env`) and that a FAISS "
            "index exists in `faiss_store/`. Run `python scripts/build_index.py` "
            "if the knowledge base hasn't been indexed yet."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Empty state: show example question chips
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, question in zip(cols, EXAMPLE_QUESTIONS):
            with col:
                st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                if st.button(question, key=f"chip_{question}", use_container_width=True):
                    run_query(rag, question, top_k, show_sources)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

    # Replay chat history
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "📚"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                render_sources(msg["sources"])

    query = st.chat_input("Ask a question about the indexed documents...")
    if query:
        run_query(rag, query, top_k, show_sources)


if __name__ == "__main__":
    main()