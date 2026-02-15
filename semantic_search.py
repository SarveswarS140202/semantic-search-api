import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

client = OpenAI()

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

# Compute embeddings once at startup
doc_embeddings = embed_texts(doc_texts)

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
            "metrics": {"latency": 0, "totalDocs": len(documents)}
        }

    query_embedding = embed_texts([req.query])[0]

    similarities = []
    for emb in doc_embeddings:
        score = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        similarities.append(score)

    similarities = normalize_scores(np.array(similarities))

    top_indices = np.argsort(similarities)[::-1][:req.k]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "id": documents[idx]["id"],
            "score": float(similarities[idx]),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

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
