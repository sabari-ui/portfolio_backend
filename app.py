import os
import atexit
import hashlib
import time
import logging
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse, quote_plus

import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from groq import Groq
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# FASTAPI INIT (FIRST — IMPORTANT)
# ---------------------------------------------------

app = FastAPI(title="Sabari Portfolio RAG API")

# ---------------------------------------------------
# MIDDLEWARE
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# BASIC ROUTES (IMMEDIATE)
# ---------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------
# ENV CONFIG
# ---------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------

groq_client = Groq(api_key=GROQ_API_KEY)
print(f"🚀 Groq model: {GROQ_MODEL}")

# ---------------------------------------------------
# EMBEDDING (LAZY LOAD)
# ---------------------------------------------------

tokenizer = None
embed_model = None
VECTOR_DIM = None

def load_embedding_model():
    global tokenizer, embed_model, VECTOR_DIM

    if tokenizer is None:
        print("🔄 Loading embedding model...")
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        embed_model = AutoModel.from_pretrained(EMBED_MODEL)
        embed_model.eval()
        VECTOR_DIM = embed_model.config.hidden_size
        print("✅ Embedding ready")

def get_embedding(text: str):
    load_embedding_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return emb

# ---------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------

def fix_db_url(url):
    parsed = urlparse(url)
    password = quote_plus(parsed.password) if parsed.password else ""
    netloc = f"{parsed.username}:{password}@{parsed.hostname}:{parsed.port or 5432}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))

DATABASE_URL = fix_db_url(DATABASE_URL)

if "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

if "connect_timeout" not in DATABASE_URL:
    DATABASE_URL += "&connect_timeout=5"

print("Connecting DB...")

connection_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL
)

print("✅ DB pool created")

@contextmanager
def get_db():
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)

atexit.register(lambda: connection_pool.closeall())

# ---------------------------------------------------
# CACHE
# ---------------------------------------------------

_cache = {}
CACHE_LIMIT = 100

def cache_key(q, k):
    return hashlib.md5(f"{q}:{k}".encode()).hexdigest()

# ---------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------

class QueryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    query: str
    k: int = 4

# ---------------------------------------------------
# VECTOR SEARCH
# ---------------------------------------------------

def vector_to_literal(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def pg_search(vec, k):
    sql = """
    SELECT chunk_id, chunk_text, document_title,
           1 - (embedding <#> %s::vector) AS score
    FROM public.chunks
    ORDER BY embedding <#> %s::vector
    LIMIT %s
    """

    vec_lit = vector_to_literal(vec)

    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (vec_lit, vec_lit, k))
            return cur.fetchall()

# ---------------------------------------------------
# GROQ GENERATION
# ---------------------------------------------------

def generate_llm(prompt):
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are Sabari's portfolio assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=400
    )

    return completion.choices[0].message.content.strip()

# ---------------------------------------------------
# MAIN RAG ENDPOINT
# ---------------------------------------------------

@app.post("/api/rag/query")
def rag_query(req: QueryRequest):

    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Query required")

    key = cache_key(q, req.k)

    if key in _cache:
        return _cache[key]

    # embedding
    emb = get_embedding(q)

    # search
    results = pg_search(emb, req.k)

    if not results:
        return {"answer": "No relevant data found", "sources": []}

    context = "\n\n".join([r["chunk_text"][:1000] for r in results])

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{q}
"""

    answer = generate_llm(prompt)

    payload = {
        "answer": answer,
        "sources": results
    }

    if len(_cache) > CACHE_LIMIT:
        _cache.pop(next(iter(_cache)))

    _cache[key] = payload

    return payload

# ---------------------------------------------------
# DEBUG ROUTES
# ---------------------------------------------------

@app.get("/test-groq")
def test_groq():
    return {"reply": generate_llm("Say hello")}

@app.get("/test-embedding")
def test_embedding():
    vec = get_embedding("hello world")
    return {"vector_size": len(vec)}
