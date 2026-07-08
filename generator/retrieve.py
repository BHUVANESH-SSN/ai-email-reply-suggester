import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from generator.config import EMBED_MODEL

_model = None


def _get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_dataset(path: str = "dataset/emails.jsonl") -> list:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def embed_texts(texts: list) -> np.ndarray:
    embedder = _get_embedder()
    return np.array(embedder.encode(list(texts), normalize_embeddings=True))


def top_k_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, k: int = 3) -> list:
    sims = corpus_embeddings @ query_embedding
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def retrieve_similar(new_subject: str, new_body: str, dataset_rows: list, k: int = 3) -> list:
    corpus_texts = [
        f"{row['incoming_email']['subject']}\n{row['incoming_email']['body']}"
        for row in dataset_rows
    ]
    corpus_embeddings = embed_texts(corpus_texts)
    query_embedding = embed_texts([f"{new_subject}\n{new_body}"])[0]
    matches = top_k_similar(query_embedding, corpus_embeddings, k=k)
    return [{"row": dataset_rows[i], "similarity": sim} for i, sim in matches]
