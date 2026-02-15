import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load local embedding model (downloads once)
model = SentenceTransformer("all-MiniLM-L6-v2")

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

# Compute embeddings once
doc_embeddings = model.encode(doc_texts)

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5


def normalize_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s - min_s == 0:
        return np.ones_like(scores) * 0.5
    return (scores - min_s) / (max_s - min_s)


def rerank(query, candidates):
    query_embedding = model.encode([query])
    candidate_texts = [c["content"] for c in candidates]
    candidate_embeddings = model.encode(candidate_texts)

    scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
    scores = normalize_scores(scores)

    for c, score in zip(candidates, scores):
        c["score"] = float(score)

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


@app.post("/search")
def search(req: SearchRequest):

    start_time = time.time()

    if not req.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {"latency": 0, "totalDocs": len(documents)}
        }

    query_embedding = model.encode([req.query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    similarities = normalize_scores(similarities)

    top_indices = np.argsort(similarities)[::-1][:req.k]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "id": documents[idx]["id"],
            "score": float(similarities[idx]),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    reranked_flag = False

    if req.rerank and len(candidates) > 0:
        candidates = rerank(req.query, candidates)
        candidates = candidates[:req.rerankK]
        reranked_flag = True
    else:
        candidates = candidates[:req.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": candidates,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
