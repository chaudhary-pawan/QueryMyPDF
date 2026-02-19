import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


class RAGChatbot:

    def __init__(self):
        # local llama3 model, no API needed
        self.llm = Ollama(
            model="llama3.2",
            num_predict=1024,   # max output tokens (increase for longer answers)
            temperature=0.1,    # low = more factual, less creative
        )

        # free offline embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

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
        """Build FAISS vector DB + document retriever."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks found. Try a different PDF.")

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        return True

    def ask(self, _pipeline, question):
        """Search context + query local model."""
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an expert research assistant. Your job is to answer questions strictly based on the provided document context below.

RULES:
- Answer ONLY using information found in the context. Do NOT use outside knowledge.
- If the context contains the answer, provide a detailed, well-structured response with specific facts, numbers, or quotes from the context.
- If the context does NOT contain enough information to answer, clearly say: "The provided document does not contain enough information to answer this question."
- Do NOT hallucinate or guess. Be precise.

---
DOCUMENT CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:
"""

        return self.llm(prompt)
