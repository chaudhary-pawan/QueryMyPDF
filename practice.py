from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("example.pdf")
docs = loader.load()
print(docs)