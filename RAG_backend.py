import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# ── Cached at module level so embeddings model loads ONCE per session ─────────
_embeddings_cache = None

def get_embeddings():
    """Load HuggingFace embeddings once and reuse (avoids re-downloading weights)."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings_cache


class RAGChatbot:

    def __init__(self):
        # ── LLM: tuned for speed ──────────────────────────────────────────
        self.llm = Ollama(
            model="llama3.2",
            num_predict=512,    # 512 is enough for most Q&A; 1024 was overkill
            temperature=0.1,    # near-zero = deterministic, no stochastic overhead
            num_ctx=2048,       # explicit context window (avoids model auto-sizing)
        )

        # ── Shared embeddings (loaded once, reused every query) ───────────
        self.embeddings = get_embeddings()
        self.last_retrieved_docs = []  # stores last query's chunks for UI inspection

    def load_pdf(self, uploaded_file):
        """Extract text from uploaded PDF."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("PDF extraction failed — file may be scanned.")

        return documents

    def create_rag_pipeline(self, documents):
        """Build FAISS vector DB + document retriever.
        
        Speed tweaks:
        - chunk_size 500 (was 700): smaller chunks → faster embedding + less context sent to LLM
        - chunk_overlap 50 (was 150): less redundant data
        - k=2 (was k=3): 2 relevant chunks is enough; fewer tokens = faster LLM response
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,     # ↑ from 500 — captures more complete passages
            chunk_overlap=100,  # ↑ from 50 — better continuity across chunk boundaries
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks found. Try a different PDF.")

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}   # ↑ from 2 — more chunks = fewer retrieval misses
        )

        return True

    def ask(self, _pipeline, question):
        """Search context + query local model. Stores retrieved docs for inspection."""
        docs = self.retriever.get_relevant_documents(question)
        self.last_retrieved_docs = docs          # ← save for UI
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""You are a precise research assistant. Answer ONLY using the document context below. If the answer is not in the context, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        return self.llm(prompt)

    def get_last_chunks(self):
        """Return retrieved chunks from the last query for display / hallucination check."""
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page":   doc.metadata.get("page", "?"),
            }
            for doc in self.last_retrieved_docs
        ]

    def ask_stream(self, _pipeline, question):
        """Streaming version — yields tokens one-by-one for live display."""
        docs = self.retriever.get_relevant_documents(question)
        self.last_retrieved_docs = docs          # ← save for UI
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""You are a precise research assistant. Answer ONLY using the document context below.
- Do NOT use your training knowledge. Do NOT invent names, titles, authors, or facts.
- If the answer is clearly present in the context, answer specifically and cite details.
- If the context does NOT contain enough information, say: "The document does not contain this information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        for chunk in self.llm.stream(prompt):
            yield chunk
