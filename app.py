"""
CodeBase QA — Streamlit UI
Full-featured interface for code Q&A with source visualization.
"""

import streamlit as st
from router import ask, list_collections, get_collection_info, ingest, delete_collection

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeBase QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Source card */
    .source-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .source-file {
        color: #67e8f9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .source-score {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Stats */
    .stat-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #67e8f9;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 CodeBase QA")
    st.markdown("---")

    # Collection selector
    st.markdown("### 📦 Repository")
    collections = list_collections()

    if not collections:
        st.warning("No repositories ingested yet. Add one below!")
        selected_collection = None
    else:
        selected_collection = st.selectbox(
            "Select collection",
            collections,
            format_func=lambda x: x.replace("repo_", "").replace("_", "-"),
        )

        # Show collection stats
        if selected_collection:
            info = get_collection_info(selected_collection)
            if "error" not in info:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div class="stat-value">{info.get("points_count", 0)}</div>'
                        f'<div class="stat-label">Chunks</div></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div class="stat-value">{info.get("status", "?")}</div>'
                        f'<div class="stat-label">Status</div></div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # Ingest new repo
    st.markdown("### ➕ Ingest New Repo")
    github_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/user/repo",
    )
    recreate = st.checkbox("Recreate if exists", value=False)

    if st.button("🚀 Ingest", use_container_width=True):
        if github_url:
            with st.spinner("Cloning, parsing, embedding, indexing..."):
                try:
                    collection = ingest(github_url, recreate=recreate)
                    st.success(f"✅ Ingested! Collection: `{collection}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Enter a GitHub URL first")

    st.markdown("---")

    top_k_search = 15
    top_k_rerank = 5

    # Danger zone
    with st.expander("🗑️ Delete Collection"):
        if collections:
            del_collection = st.selectbox(
                "Collection to delete",
                collections,
                key="del_select",
                format_func=lambda x: x.replace("repo_", "").replace("_", "-"),
            )
            if st.button("Delete", type="secondary", use_container_width=True):
                if delete_collection(del_collection):
                    st.success(f"Deleted `{del_collection}`")
                    st.rerun()
                else:
                    st.error("Failed to delete")


# ── Main Area ───────────────────────────────────────────────────

# Header
st.markdown(
    '<div class="main-header">'
    '<h1>💬 CodeBase QA</h1>'
    '<p>Ask natural language questions about any codebase. '
    'Powered by Tree-sitter parsing, hybrid search, cross-encoder reranking, and Llama 3.3.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Query input
question = st.text_area(
    "Ask a question about the code",
    placeholder="e.g., How is authentication implemented? / What does the login function do? / Explain the middleware pipeline...",
    height=80,
    label_visibility="collapsed",
)

col_ask, col_info = st.columns([1, 3])
with col_ask:
    ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
with col_info:
    if selected_collection:
        st.caption(
            f"Searching in **{selected_collection.replace('repo_', '').replace('_', '-')}** "
            f"| Top-{top_k_search} → Rerank → Top-{top_k_rerank}"
        )

# Process query
if ask_clicked and question and selected_collection:
    with st.spinner("🔍 Searching → 🏆 Reranking → 🤖 Generating..."):
        result = ask(
            question=question,
            collection_name=selected_collection,
            top_k_search=top_k_search,
            top_k_rerank=top_k_rerank,
        )

    # Display answer
    st.markdown("### 📝 Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Model: `{result['model']}`")

    # Display sources
    if result["sources"]:
        st.markdown("### 📎 Sources")

        for i, src in enumerate(result["sources"], 1):
            score_pct = int(src["score"] * 100) if src["score"] <= 1 else int(src["score"])

            with st.expander(
                f"📄 {src['file']} → `{src['name']}()` | "
                f"Lines {src['lines']} | "
                f"Score: {src['score']:.3f}"
            ):
                st.markdown(
                    f"**Type**: `{src['type']}` | "
                    f"**Language**: `{src['language']}` | "
                    f"**Lines**: `{src['lines']}`"
                )
                st.code(src["code"], language=src["language"])

elif ask_clicked and not selected_collection:
    st.warning("⚠️ Please select a repository collection first.")
elif ask_clicked and not question:
    st.warning("⚠️ Please enter a question.")

# Empty state
if not ask_clicked:
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; padding: 3rem; opacity: 0.5;">'
        '<p style="font-size: 3rem;">🔍</p>'
        '<p>Select a repository and ask a question to get started.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
