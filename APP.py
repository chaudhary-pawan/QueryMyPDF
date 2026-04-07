import uuid
import streamlit as st
from RAG_backend import (
    ingest_pdf,
    chatbot,
    retrieve_all_threads,
    thread_has_document,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QueryMyPDF — AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 18px !important;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    letter-spacing: -1px;
}
.hero p {
    color: #94a3b8;
    font-size: 2.35rem;
    font-weight: 300;
}

/* ── Upload card ── */
.upload-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ── Status pills ── */
.pill-success {
    display: inline-block;
    background: linear-gradient(90deg, #059669, #10b981);
    color: white;
    font-weight: 600;
    font-size: 1.50rem;
    padding: 0.45rem 1.2rem;
    border-radius: 999px;
    margin: 0.8rem 0;
}
.pill-info {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 600;
    font-size: 1.50rem;
    padding: 0.45rem 1.2rem;
    border-radius: 999px;
    margin: 0.8rem 0;
}

/* ── Chat bubbles ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    margin: 1.5rem 0;
}
.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 1.1rem 1.5rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 1.50rem;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.bubble-ai {
    align-self: flex-start;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e2e8f0;
    padding: 1.2rem 1.6rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 82%;
    font-size: 1.50rem;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    line-height: 1.85;
}
.bubble-label {
    font-size: 1.95rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.4rem;
}

/* ── Input area ── */
.stTextInput input {
    background: rgba(255,255,255,0.9) !important;
    border: 1.5px solid rgba(167,139,250,0.35) !important;
    border-radius: 14px !important;
    color: #000000 !important;
    padding: 0.9rem 1.2rem !important;
    font-size: 1.50rem !important;
    transition: border-color 0.2s;
}
.stTextInput input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}
.stTextInput > div > div > input::placeholder { color: #64748b !important; font-size: 1.1rem !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 2.4rem !important;
    font-weight: 600 !important;
    font-size: 2.15rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.45) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.6) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Section label ── */
.section-label {
    color: #94a3b8;
    font-size: 2.05rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #475569;
    font-size: 2.1rem;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📄 QueryMyPDF</h1>
    <p>Drop any PDF. Ask anything. Get instant AI-powered answers.</p>
</div>
""", unsafe_allow_html=True)

# ─── Init session state ───────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "pdf_meta" not in st.session_state:
    st.session_state.pdf_meta = {}

thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}

# ─── Layout: two columns ─────────────────────────────────────────────────────
left, right = st.columns([1.1, 1.9], gap="large")

with left:
    st.markdown('<div class="section-label">📂 Upload Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop a PDF here",
        type="pdf",
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f'<div class="pill-info">📎 {uploaded_file.name}</div>', unsafe_allow_html=True)

        if st.button("⚡ Build Knowledge Base"):
            with st.spinner("📘 Reading & indexing PDF..."):
                meta = ingest_pdf(
                    file_bytes=uploaded_file.read(),
                    thread_id=thread_id,
                    filename=uploaded_file.name,
                )
            st.session_state.pdf_ready = True
            st.session_state.pdf_meta = meta
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown('<p style="color:#475569;font-size:1.50rem;margin-top:0.5rem;">Upload a PDF file to get started.</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.pdf_ready:
        meta = st.session_state.pdf_meta
        st.markdown('<div class="pill-success">✅ Knowledge base ready</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#64748b;font-size:1.1rem;">'
            f'📄 {meta.get("filename","?")} &nbsp;·&nbsp; '
            f'{meta.get("documents","?")} pages &nbsp;·&nbsp; '
            f'{meta.get("chunks","?")} chunks</p>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Info card
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        color: #64748b;
        font-size: 1.05rem;
        line-height: 2;
    ">
        🧠 <b style="color:#94a3b8">Gemini 2.5 Flash</b><br>
        📦 <b style="color:#94a3b8">FAISS Vector Search</b><br>
        🔗 <b style="color:#94a3b8">LangGraph Agent</b><br>
        🌐 <b style="color:#94a3b8">Google Embeddings</b>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">💬 Chat with your documents</div>', unsafe_allow_html=True)

    # ── Chat history ──
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for turn in st.session_state.chat_history:
            st.markdown(f"""
            <div class="bubble-user">
                <div class="bubble-label">You</div>
                {turn['question']}
            </div>
            <div class="bubble-ai">
                <div class="bubble-label">🤖 AI Answer</div>
                {turn['answer']}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding: 4rem 2rem;
            color: #334155;
            border: 2px dashed rgba(255,255,255,0.06);
            border-radius: 20px;
        ">
            <div style="font-size:2rem;margin-bottom:1rem;">🔍</div>
            <div style="font-size:1rem;color:#475569;">Your answers will appear here.<br>Upload a PDF and ask away!</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input area ──
    if st.session_state.pdf_ready:
        st.markdown("<br>", unsafe_allow_html=True)
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the main contribution of this paper?",
            label_visibility="collapsed"
        )
        ask_btn = st.button("🚀 Ask AI")

        if ask_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("🤔 Thinking..."):
                    # Inject thread_id into tool call via system context
                    result = chatbot.invoke(
                        {"messages": [HumanMessage(content=question)]},
                        config=config,
                    )
                    raw_content = result["messages"][-1].content
                    if isinstance(raw_content, list):
                        answer = "".join(
                            block["text"] for block in raw_content if isinstance(block, dict) and "text" in block
                        )
                    else:
                        answer = raw_content

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                })
                st.rerun()
    else:
        st.markdown("""
        <div style="
            text-align:center;
            padding: 1rem;
            color: #475569;
            font-size: 0.9rem;
        ">
            ← Upload a PDF and build the knowledge base to start chatting.
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit · Gemini 2.5 Flash · FAISS · LangGraph
</div>
""", unsafe_allow_html=True)
