import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


class RAGChatbot:

    def __init__(self):
        # local llama3 model, no API needed
        self.llm = Ollama(model="llama3")

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
You are a helpful assistant.
Here is retrieved context from PDF:

{context}

Question: {question}

Provide a clear answer:
"""

        return self.llm(prompt)
