from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH = "data"
FAISS_INDEX_PATH = "faiss_index"

def main():
    documents = load_pdf_file(DATA_PATH)
    texts = text_split(documents)

    embeddings = download_hugging_face_embeddings()

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    print("✅ FAISS index created successfully")

if __name__ == "__main__":
    main()
