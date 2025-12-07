import streamlit as st
from RAG_backend import RAGChatbot

st.set_page_config(page_title="Offline RAG Chatbot", layout="wide")
st.title("📚 Offline PDF-RAG Chatbot (LLaMA-3 + FAISS)")

st.write("Upload PDFs and chat with your documents — no API required!")

# Upload PDFs
uploaded_files = st.file_uploader(
    "📂 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("Please upload at least one PDF.")
    st.stop()

rag = RAGChatbot()

with st.spinner("📘 Reading PDFs..."):
    all_docs = []
    for f in uploaded_files:
        all_docs.extend(rag.load_pdf(f))

with st.spinner("⚙️ Indexing knowledge base..."):
    pipeline_ready = rag.create_rag_pipeline(all_docs)

st.success("🎉 RAG is ready! Ask anything from the documents below:")

question = st.text_input("💬 Ask a question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer = rag.ask(pipeline_ready, question)

        st.markdown("### 🧠 Answer:")
        st.write(answer)
