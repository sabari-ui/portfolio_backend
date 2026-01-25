# app.py

import os
import hashlib
import time
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse, quote_plus

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import RealDictCursor

from groq import Groq
from sentence_transformers import SentenceTransformer


# ------------------------------------------------
# ENV LOAD
# ------------------------------------------------

load_dotenv()

# ------------------------------------------------
# FASTAPI INIT
# ------------------------------------------------

app = FastAPI(title="Sabari Portfolio RAG API")

# ------------------------------------------------
# MIDDLEWARE
# ------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# BASIC ROUTES
# ------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------
# ENV CONFIG
# ------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

# ------------------------------------------------
# GROQ CLIENT
# ------------------------------------------------

groq_client = Groq(api_key=GROQ_API_KEY)
print(f"🚀 Groq model: {GROQ_MODEL}")

# ------------------------------------------------
# EMBEDDING (LAZY LOAD)
# ------------------------------------------------

embedder = None

def load_embedding_model():
    global embedder
    if embedder is None:
        print("🔄 Loading embedding model...")
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        print("✅ Embedding model ready")

def get_embedding(text: str):
    load_embedding_model()
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()

# ------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------

def fix_db_url(url):
    parsed = urlparse(url)
    password = quote_plus(parsed.password) if parsed.password else ""
    netloc = f"{parsed.username}:{password}@{parsed.hostname}:{parsed.port or 5432}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))

DATABASE_URL = fix_db_url(DATABASE_URL)

if "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

if "connect_timeout" not in DATABASE_URL:
    DATABASE_URL += "&connect_timeout=10"

print("✅ Database URL prepared")

# ------------------------------------------------
# SAFE CONNECTION HANDLER (NO POOLER CONFLICT)
# ------------------------------------------------

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    finally:
        if conn:
            conn.close()

# ------------------------------------------------
# CACHE SYSTEM
# ------------------------------------------------

_answer_cache = {}
_CACHE_LIMIT = 100

def _cache_key(query, k):
    return hashlib.md5(f"{query}:{k}".encode()).hexdigest()

def _get_cached(key):
    return _answer_cache.get(key)

def _store_cache(key, payload):
    if len(_answer_cache) >= _CACHE_LIMIT:
        _answer_cache.pop(next(iter(_answer_cache)))
    _answer_cache[key] = payload

@app.get("/cache/stats")
def cache_stats():
    return {
        "size": len(_answer_cache),
        "max_size": _CACHE_LIMIT
    }

@app.get("/cache/clear")
def clear_cache():
    _answer_cache.clear()
    return {"status": "cache cleared"}

# ------------------------------------------------
# REQUEST MODEL
# ------------------------------------------------

class QueryRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    query: str
    k: int = 4

# ------------------------------------------------
# VECTOR SEARCH
# ------------------------------------------------

def vector_to_literal(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def pg_search(vec, k):

    sql = """
    SELECT chunk_id,
           chunk_text,
           document_title,
           1 - (embedding <#> %s::vector) AS score
    FROM public.chunks
    ORDER BY embedding <#> %s::vector
    LIMIT %s
    """

    vec_lit = vector_to_literal(vec)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (vec_lit, vec_lit, k))
            return cur.fetchall()

# ------------------------------------------------
# GROQ GENERATION
# ------------------------------------------------

def generate_llm(prompt):

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are Sabari's portfolio assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )

    return completion.choices[0].message.content.strip()

# ------------------------------------------------
# RAG ENDPOINT
# ------------------------------------------------

@app.post("/api/rag/query")
def rag_query(req: QueryRequest):

    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Query required")

    cache_key = _cache_key(q, req.k)
    cached = _get_cached(cache_key)

    if cached:
        return cached

    start = time.time()

    emb = get_embedding(q)
    results = pg_search(emb, req.k)

    if not results:
        return {"answer": "No relevant data found", "sources": []}

    context = "\n\n".join([r["chunk_text"][:1200] for r in results])

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{q}
"""

    answer = generate_llm(prompt)

    payload = {
        "answer": answer,
        "sources": results,
        "timings": {
            "total_seconds": round(time.time() - start, 2),
            "cached": False
        }
    }

    _store_cache(cache_key, payload)

    return payload

# ------------------------------------------------
# DEBUG ROUTES
# ------------------------------------------------

@app.get("/test-groq")
def test_groq():
    return {"reply": generate_llm("Say hello in one sentence")}

@app.get("/test-embedding")
def test_embedding():
    vec = get_embedding("hello world")
    return {"vector_size": len(vec)}

@app.get("/health/detailed")
def detailed_health():

    db_status = "ok"

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "embedding_model": EMBED_MODEL,
        "llm_model": GROQ_MODEL,
        "embedding_dim": embedder.get_sentence_embedding_dimension() if embedder else None
    }
