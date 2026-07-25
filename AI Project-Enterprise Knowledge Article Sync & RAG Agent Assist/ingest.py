import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. Import HuggingFace Embeddings instead of OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "./knowledge_articles"
CHROMA_DIR = "./chroma_db"

def load_all_documents():
    documents = []
    path = Path(DATA_DIR)
    
    for pdf_file in path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
        
    for md_file in path.glob("*.md"):
        loader = TextLoader(str(md_file))
        documents.extend(loader.load())
        
    return documents

def build_vector_database():
    docs = load_all_documents()
    if not docs:
        print("No documents found in 'knowledge_articles/'!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 2. Use a free, lightweight open-source embedding model that runs locally on your CPU
    print("Loading free local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("✅ Vector Database successfully built locally for FREE!")

if __name__ == "__main__":
    build_vector_database()