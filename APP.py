import uuid
import streamlit as st
import streamlit.components.v1 as components
from RAG_backend import (
    ingest_pdf,
    chatbot,
    thread_has_document,
    thread_document_metadata,
)
from langchain_core.messages import HumanMessage

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QueryMyPDF — AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #08081a !important;
}

#MainMenu, footer { display: none !important; }
header { background: transparent !important; }
[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 99999 !important;
}
[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
[data-testid="stHeader"] [data-testid="collapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
}

.stApp {
    background: #08081a !important;
    background-image:
        radial-gradient(ellipse at 10% 40%, rgba(109,40,217,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 70%, rgba(37,99,235,0.13) 0%, transparent 55%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(6,6,20,0.95) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
.s-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.55rem; font-weight: 700;
    background: linear-gradient(135deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding-bottom: 1.2rem;
}
.s-section {
    font-size: 0.67rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #475569; margin: 1.4rem 0 0.45rem;
}
.s-meta { font-size: 0.82rem; color: #64748b; line-height: 1.75; }
.pill-green {
    display: inline-block;
    background: rgba(16,185,129,0.1); color: #34d399;
    border: 1px solid rgba(52,211,153,0.22); border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; padding: 0.18rem 0.65rem; margin-bottom: 0.4rem;
}
.pill-violet {
    display: inline-block;
    background: rgba(139,92,246,0.1); color: #a78bfa;
    border: 1px solid rgba(167,139,250,0.22); border-radius: 999px;
    font-size: 0.7rem; font-weight: 600; padding: 0.18rem 0.65rem; margin-bottom: 0.5rem;
}
.s-engine { font-size: 0.77rem; color: #334155; line-height: 2.0; }
.s-engine b { color: #475569; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(139,92,246,0.28) !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.55rem 1rem !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
    font-size: 0.88rem !important; transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(109,40,217,0.35) !important; width: 100% !important;
    min-height: 42px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(109,40,217,0.55) !important;
}

/* ── Pause / Stop button override ── */
.stop-col .stButton > button {
    background: rgba(239,68,68,0.06) !important;
    border: 1px solid rgba(239,68,68,0.28) !important;
    color: #f87171 !important; border-radius: 14px !important;
    padding: 0.6rem 0.9rem !important; font-size: 1.0rem !important;
    font-weight: 400 !important; box-shadow: none !important; width: auto !important;
    min-height: 48px !important;
}
.stop-col .stButton > button:hover {
    background: rgba(239,68,68,0.14) !important;
    border-color: rgba(239,68,68,0.5) !important;
    transform: scale(1.06) !important; box-shadow: none !important;
}

/* ── Chat container (applied to the st.container box) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
}

/* ── Chat bubbles ── */
.bubble-user-wrap { display: flex; justify-content: flex-end; margin: 0.5rem 0; }
.bubble-ai-wrap   { display: flex; justify-content: flex-start; margin: 0.5rem 0; }

.bubble-user {
    max-width: 70%; background: linear-gradient(135deg, #6d28d9, #4f46e5);
    color: #fff; padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px; font-size: 0.92rem;
    line-height: 1.6; box-shadow: 0 6px 24px rgba(109,40,217,0.3);
    word-break: break-word;
}
.bubble-ai {
    max-width: 78%; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08); color: #cbd5e1;
    padding: 1rem 1.3rem; border-radius: 18px 18px 18px 4px;
    font-size: 0.92rem; line-height: 1.75;
    backdrop-filter: blur(10px); box-shadow: 0 6px 30px rgba(0,0,0,0.35);
    word-break: break-word;
}
.bubble-ai pre, .bubble-ai code, .bubble-user pre, .bubble-user code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}
.bubble-err {
    max-width: 78%; background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.22); color: #fca5a5;
    padding: 0.9rem 1.2rem; border-radius: 18px 18px 18px 4px;
    font-size: 0.89rem; line-height: 1.6;
    word-break: break-word;
}
.lbl { font-family:'Outfit',sans-serif; font-size:0.62rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase; opacity:0.6; margin-bottom:0.3rem; }
.lbl-ai { color:#818cf8; opacity:1; display:flex; align-items:center; gap:0.35rem; }
.ai-dot { width:6px;height:6px;border-radius:50%;background:#818cf8;
    display:inline-block; animation: dotPulse 2s ease-in-out infinite; }
@keyframes dotPulse {
    0%,100%{box-shadow:0 0 0 0 rgba(129,140,248,0.7)}
    50%{box-shadow:0 0 0 5px rgba(129,140,248,0)} }

/* Empty state */
.empty-chat { text-align:center; padding: 5rem 1rem; }
.empty-icon { font-size:2.8rem; animation: flt 4s ease-in-out infinite; display:block; margin-bottom:1rem;}
@keyframes flt{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
.empty-title { font-family:'Outfit',sans-serif; font-size:1.8rem; font-weight:700;
    color:#1e293b; margin-bottom:0.5rem; }
.empty-title.rdy { color:#e2e8f0; }
.empty-sub { font-size:0.9rem; color:#334155; line-height:1.7; font-weight:300; }
.empty-sub.rdy { color:#64748b; }

/* ── Top header ── */
.top-header {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.9rem 0 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.5rem;
}
.top-header-title { font-family:'Outfit',sans-serif; font-size:0.78rem; font-weight:600;
    color:#475569; letter-spacing:0.08em; text-transform:uppercase; }
.status-badge { display:flex; align-items:center; gap:0.35rem; font-size:0.7rem; color:#34d399; }
.sdot { width:6px;height:6px;border-radius:50%;background:#34d399;
    animation:dotPulse 2.5s ease-in-out infinite; }

/* ── Input bar ── */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottomBlockContainer"] > div > div {
    background: transparent !important;
    box-shadow: none !important; border: none !important;
}
[data-testid="stChatInput"] { background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea {
    background: rgba(10,14,32,0.82) !important;
    border: 1px solid rgba(99,102,241,0.28) !important;
    color: #e2e8f0 !important; border-radius: 20px !important;
    padding: 1rem 1.5rem !important; font-size: 0.92rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
    resize: none !important; min-height: 54px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.12), 0 0 24px rgba(109,40,217,0.1) !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#374151 !important; font-style:italic !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg,#6d28d9,#4f46e5) !important;
    border-radius: 12px !important; border: none !important; color: white !important;
    box-shadow: 0 4px 14px rgba(109,40,217,0.4) !important; transition: all 0.2s !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.1) !important; box-shadow: 0 6px 20px rgba(109,40,217,0.65) !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; }
.block-container { padding: 1rem 2rem 0 2rem !important; max-width: 100% !important; }

/* ── Mobile & Tablet Responsiveness ── */
@media (max-width: 768px) {
    .block-container {
        padding: 0.5rem 0.6rem 0 0.6rem !important;
    }
    .top-header {
        padding: 0.4rem 0 0.4rem !important;
    }
    .empty-chat {
        padding: 2.2rem 0.5rem !important;
    }
    .empty-icon {
        font-size: 2.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    .empty-title {
        font-size: 1.35rem !important;
    }
    .empty-sub {
        font-size: 0.82rem !important;
    }
    .bubble-user {
        max-width: 88% !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.88rem !important;
    }
    .bubble-ai, .bubble-err {
        max-width: 92% !important;
        padding: 0.85rem 1rem !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important; /* Prevents auto-zoom on mobile safari/chrome */
        padding: 0.85rem 1.1rem !important;
        min-height: 48px !important;
    }
    .stButton > button {
        min-height: 44px !important; /* Touch target minimum height */
    }
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 0.4rem !important;
        flex-wrap: nowrap !important;
    }
    [data-testid="stColumn"]:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    [data-testid="stColumn"]:last-child {
        flex: 0 0 auto !important;
        width: auto !important;
    }
    .stop-col .stButton > button {
        padding: 0.5rem 0.8rem !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
}

@media (max-height: 700px) {
    .empty-chat {
        padding: 1.5rem 0.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ─── State ──────────────────────────────────────────────────────────────────
if "thread_id"    not in st.session_state: st.session_state.thread_id    = str(uuid.uuid4())
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "pdf_ready"    not in st.session_state: st.session_state.pdf_ready    = False
if "pdf_meta"     not in st.session_state: st.session_state.pdf_meta     = {}
if "stop_stream"  not in st.session_state: st.session_state.stop_stream  = False

thread_id = st.session_state.thread_id
config    = {"configurable": {"thread_id": thread_id}}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="s-title">QueryMyPDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-section">📂 Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        st.markdown(f'<div class="pill-violet">📎 {uploaded_file.name}</div>', unsafe_allow_html=True)
        if st.button("⚡ Build Knowledge Base"):
            with st.spinner("Indexing..."):
                try:
                    meta = ingest_pdf(
                        file_bytes=uploaded_file.read(),
                        thread_id=thread_id,
                        filename=uploaded_file.name,
                    )
                    st.session_state.pdf_ready    = True
                    st.session_state.pdf_meta     = meta
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if st.session_state.pdf_ready:
        meta = st.session_state.pdf_meta
        st.markdown('<div class="s-section">Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="pill-green">&#x2714; Active</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="s-meta">📄 {meta.get("filename","?")}<br>'
            f'📑 {meta.get("documents","?")} pages &nbsp;&#183;&nbsp; '
            f'🧩 {meta.get("chunks","?")} chunks</div>',
            unsafe_allow_html=True
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div class="s-engine">
        <b>Engine</b>&nbsp; Gemini 2.5 Flash<br>
        <b>Retrieval</b>&nbsp; BM25 + FAISS<br>
        <b>Embeddings</b>&nbsp; HuggingFace<br>
        <b>Agent</b>&nbsp; LangGraph
    </div>""", unsafe_allow_html=True)


# ─── Main area ───────────────────────────────────────────────────────────────
ready   = st.session_state.pdf_ready
history = st.session_state.chat_history

# Header bar
if ready:
    st.markdown(
        '<div class="top-header">'
        '<span class="top-header-title">AI Chat</span>'
        '<span class="status-badge"><span class="sdot"></span>Connected</span>'
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="top-header"><span class="top-header-title">AI Chat</span></div>',
        unsafe_allow_html=True
    )

# Track pending question across reruns so user bubble renders INSIDE the container
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

# ── Fixed-height scrollable chat container ──
chat_box = st.container(height=500, border=False)

with chat_box:
    if not ready:
        st.markdown("""
        <div class="empty-chat">
            <span class="empty-icon">📄</span>
            <div class="empty-title">Welcome.</div>
            <div class="empty-sub">Upload a PDF in the sidebar to get started.<br>Ask it anything — I will dig through every page.</div>
        </div>""", unsafe_allow_html=True)
    elif not history and not st.session_state.pending_q:
        st.markdown("""
        <div class="empty-chat">
            <span class="empty-icon">🧠</span>
            <div class="empty-title rdy">Knowledge Base Online</div>
            <div class="empty-sub rdy">I have read and indexed your document.<br>What would you like to know?</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Render all committed history
        for turn in history:
            q      = turn["question"]
            a      = turn["answer"]
            is_err = turn.get("is_error", False)
            b_cls  = "bubble-err" if is_err else "bubble-ai"
            lbl_ai = "" if is_err else '<div class="lbl lbl-ai"><span class="ai-dot"></span>Assistant</div>'
            st.markdown(
                f'<div class="bubble-user-wrap"><div class="bubble-user"><div class="lbl">You</div>{q}</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="bubble-ai-wrap"><div class="{b_cls}">{lbl_ai}{a}</div></div>',
                unsafe_allow_html=True
            )

        # If there is a pending question, show its user bubble + a live AI slot
        if st.session_state.pending_q:
            pq = st.session_state.pending_q
            st.markdown(
                f'<div class="bubble-user-wrap"><div class="bubble-user"><div class="lbl">You</div>{pq}</div></div>',
                unsafe_allow_html=True
            )
            ai_slot = st.empty()


# ─── Input bar ───────────────────────────────────────────────────────────────
if ready:
    inp_col, stop_col = st.columns([11, 1])

    with inp_col:
        question = st.chat_input("Ask a question about your document...", key="qinput")

    with stop_col:
        st.markdown('<div class="stop-col">', unsafe_allow_html=True)
        if st.button("⏸", key="stop_btn", help="Stop generation"):
            st.session_state.stop_stream = True
            st.session_state.pending_q   = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 1: New question submitted → store as pending and rerun to show bubble ──
    if question and not st.session_state.pending_q:
        st.session_state.pending_q  = question
        st.session_state.stop_stream = False
        st.rerun()

    # ── Step 2: Pending question exists → stream AI response inside the container ──
    if st.session_state.pending_q:
        pq           = st.session_state.pending_q
        full_response = ""
        is_err        = False

        try:
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=pq)]},
                config=config,
                stream_mode="messages"
            ):
                if st.session_state.get("stop_stream"):
                    break
                if metadata.get("langgraph_node") == "chat_node":
                    token = ""
                    if isinstance(chunk.content, str):
                        token = chunk.content
                    elif isinstance(chunk.content, list):
                        for block in chunk.content:
                            if isinstance(block, dict) and "text" in block:
                                token += block["text"]
                    if token:
                        full_response += token
                        ai_slot.markdown(
                            f'<div class="bubble-ai-wrap"><div class="bubble-ai">'
                            f'<div class="lbl lbl-ai"><span class="ai-dot"></span>Assistant</div>'
                            f'{full_response}&#9646;</div></div>',
                            unsafe_allow_html=True
                        )

            # Remove cursor
            if full_response:
                ai_slot.markdown(
                    f'<div class="bubble-ai-wrap"><div class="bubble-ai">'
                    f'<div class="lbl lbl-ai"><span class="ai-dot"></span>Assistant</div>'
                    f'{full_response}</div></div>',
                    unsafe_allow_html=True
                )

            # Fallback invoke if stream returned nothing
            if not full_response and not st.session_state.get("stop_stream"):
                result = chatbot.invoke(
                    {"messages": [HumanMessage(content=pq)]}, config=config
                )
                raw = result["messages"][-1].content
                full_response = (
                    "".join(b["text"] for b in raw if isinstance(b, dict) and "text" in b)
                    if isinstance(raw, list) else raw
                )
                ai_slot.markdown(
                    f'<div class="bubble-ai-wrap"><div class="bubble-ai">'
                    f'<div class="lbl lbl-ai"><span class="ai-dot"></span>Assistant</div>'
                    f'{full_response}</div></div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                full_response = "The AI is rate-limited (Gemini quota exceeded). Please wait and try again."
            elif "503" in err:
                full_response = "The AI service is temporarily unavailable. Retry in a few seconds."
            else:
                full_response = "An error occurred. Please try again."
            is_err = True
            ai_slot.markdown(
                f'<div class="bubble-ai-wrap"><div class="bubble-err">{full_response}</div></div>',
                unsafe_allow_html=True
            )

        # Commit to history, clear pending, rerun to refresh container
        if full_response:
            st.session_state.chat_history.append({
                "question": pq,
                "answer":   full_response,
                "is_error": is_err,
            })
        st.session_state.pending_q = None
        st.rerun()
