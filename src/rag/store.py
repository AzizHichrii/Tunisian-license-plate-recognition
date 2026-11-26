"""Shared RAG helpers: database connection + embedding model."""
import os
from functools import lru_cache

import psycopg
import torch
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tlpr:tlpr_dev@localhost:5433/tlpr")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)   # lets psycopg send/receive numpy vectors
    return conn


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.max_seq_length = 512   # our chunks are short; saves GPU memory
    if device == "cuda":
        model.half()             # half precision: 2x less VRAM
    return model


def embed(texts: list[str], batch_size: int = 16):
    return get_embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 100)