from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
index_file = "vectorstore/faiss.index"
text_file = "vectorstore/text.pkl"

def embed_and_store(text_chunks):
    if not os.path.exists("vectorstore"):
        os.mkdir("vectorstore")

    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_file)

    with open(text_file, "wb") as f:
        pickle.dump(text_chunks, f)

def load_vectorstore():
    index = faiss.read_index(index_file)
    with open(text_file, "rb") as f:
        text_chunks = pickle.load(f)
    return index, text_chunks
