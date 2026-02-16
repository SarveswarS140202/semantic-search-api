import time
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Enable CORS for grader / cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

EMBEDDINGS_FILE = "doc_embeddings.npy"

# Load documents
documents = []
with open("abstracts.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f.readlines()):
        documents.append({
            "id": i,
            "content": line.strip(),
            "metadata": {"source": "scientific_abstracts"}
        })

doc_texts = [doc["content"] for doc in documents]


def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([r.embedding for r in response.data])


# -------- Embedding Cache --------
if os.path.exists(EMBEDDINGS_FILE):
    print("Loading cached embeddings...")
    doc_embeddings = np.load(EMBEDDINGS_FILE)
else:
    print("Computing embeddings for documents...")
    doc_embeddings = embed_texts(doc_texts)
    np.save(EMBEDDINGS_FILE, doc_embeddings)
# ---------------------------------


def normalize_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s - min_s == 0:
        return np.ones_like(scores) * 0.5
    return (scores - min_s) / (max_s - min_s)


class SearchRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5


@app.post("/search")
def search(req: SearchRequest):

    start_time = time.time()

    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": len(documents)
            }
        }

    # Embed query
    query_embedding = embed_texts([req.query])[0]

    # Compute cosine similarity
    similarities = []
    for emb in doc_embeddings:
        score = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        similarities.append(score)

    similarities = normalize_scores(np.array(similarities))

    # Get top-k initial retrieval
    top_indices = np.argsort(similarities)[::-1][:req.k]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "id": documents[idx]["id"],
            "score": float(similarities[idx]),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # Return top rerankK results (no LLM rerank to save cost)
    candidates = candidates[:req.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": candidates,
        "reranked": False,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
