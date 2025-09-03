from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load resumes
data_path = "data"
all_docs = []
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
docs = splitter.split_documents(all_docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Chroma
db = Chroma.from_documents(docs, embeddings, persist_directory="db")
db.persist()
# print(f"Ingested {len(docs)} chunks")
