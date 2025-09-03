from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing DB
db = Chroma(persist_directory="db", embedding_function=embedding)

# Delete everything
db.delete_collection()

print("Vector DB cleared. You can now re-run ingest.py")
