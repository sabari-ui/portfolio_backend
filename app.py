# app.py
import os
import atexit
import requests
import hashlib
from contextlib import contextmanager
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import torch
from transformers import AutoModel, AutoTokenizer



load_dotenv()

# ---------------------------
# ENV VARIABLES
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-instant")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY required in .env file")

# ---------------------------
# INIT CLIENTS
# ---------------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# Load Nomic embedding model
print(f"🚀 Loading Nomic embedding model: {EMBED_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
embed_model.eval()

VECTOR_DIM = embed_model.config.hidden_size
print(f"✅ Nomic model loaded. Embedding dimension = {VECTOR_DIM}")
print(f"🚀 Groq LLM model: {GROQ_MODEL}")

# Get database connection - can use DATABASE_URL directly or construct from components
from urllib.parse import urlparse, urlunparse, quote_plus

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

def validate_and_fix_database_url(url):
    """Validate and fix DATABASE_URL format."""
    if not url:
        return None
    
    # Check if it's a valid PostgreSQL URL
    if not url.startswith(("postgresql://", "postgres://")):
        return None
    
    try:
        parsed = urlparse(url)
        
        # Check if password is missing or malformed
        if not parsed.password and "@" in url:
            # Password might be missing
            return None
        
        # Ensure it has all required components
        if not parsed.hostname:
            return None
        
        # Reconstruct URL to ensure proper format
        if parsed.password:
            # URL encode password in case it has special characters
            password_encoded = quote_plus(parsed.password)
            netloc = f"{parsed.username}:{password_encoded}@{parsed.hostname}"
        else:
            netloc = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname
        
        if parsed.port:
            netloc += f":{parsed.port}"
        
        fixed_url = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path or "/postgres",
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        
        return fixed_url
    except Exception as e:
        print(f"Warning: Could not parse DATABASE_URL: {e}")
        return None

if DATABASE_URL:
    # Validate and fix the URL
    fixed_url = validate_and_fix_database_url(DATABASE_URL)
    if fixed_url:
        DATABASE_URL = fixed_url
    else:
        print(f"Warning: DATABASE_URL format may be incorrect: {DATABASE_URL[:50]}...")
        print("Expected format: postgresql://user:password@host:port/database")

if not DATABASE_URL or not DATABASE_URL.startswith(("postgresql://", "postgres://")):
    # Try to construct from Supabase components if available
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_db_password = os.getenv("SUPABASE_DB_PASSWORD")
    supabase_db_host = os.getenv("SUPABASE_DB_HOST")
    supabase_db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    supabase_db_user = os.getenv("SUPABASE_DB_USER", "postgres")
    
    if supabase_db_host and supabase_db_password:
        db_password_encoded = quote_plus(supabase_db_password)
        DATABASE_URL = f"postgresql://{supabase_db_user}:{db_password_encoded}@{supabase_db_host}:5432/{supabase_db_name}"
        print(f"Constructed DATABASE_URL from Supabase components")
    elif supabase_url:
        raise SystemExit(
            "DATABASE_URL not found or invalid in .env file.\n\n"
            "SUPABASE_URL is set, but you also need DATABASE_URL (PostgreSQL connection string).\n\n"
            "To get DATABASE_URL:\n"
            "1. Go to Supabase Dashboard > Settings > Database\n"
            "2. Find 'Connection string' section\n"
            "3. Copy the 'URI' connection string (starts with postgresql://)\n"
            "4. Add it to your .env file as: DATABASE_URL=postgresql://user:password@host:port/database\n\n"
            "Format should be: postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres\n\n"
            "Alternatively, set these in .env:\n"
            "  SUPABASE_DB_HOST=db.xxx.supabase.co\n"
            "  SUPABASE_DB_PASSWORD=your_db_password\n"
            "  SUPABASE_DB_NAME=postgres\n"
            "  SUPABASE_DB_USER=postgres"
        )
    else:
        raise SystemExit(
            "DATABASE_URL required in .env file.\n\n"
            "Get it from: Supabase Dashboard > Settings > Database > Connection string (URI)\n"
            "Format: postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres"
        )

# Vector dimension - will be set based on actual model dimension
# Allow all origins for local development (file:// and localhost)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]

# Connection pool for thread-safe database access
try:
    # Validate URL format before using it
    parsed = urlparse(DATABASE_URL)
    if not parsed.hostname:
        raise ValueError("DATABASE_URL missing hostname")
    if not parsed.username:
        raise ValueError("DATABASE_URL missing username")
    
    # Ensure SSL for Supabase
    if "sslmode" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        DATABASE_URL = f"{DATABASE_URL}{separator}sslmode=require"
    
    # Debug: show masked URL (hide password)
    safe_url = DATABASE_URL
    if "@" in safe_url:
        parts = safe_url.split("@")
        if ":" in parts[0]:
            user_pass = parts[0].split(":", 1)
            safe_url = f"{user_pass[0]}:***@{parts[1]}"
    print(f"Connecting to database: {safe_url}")
    
    # Note: Python's socket.getaddrinfo might fail even if nslookup works
    # This is common on Windows with IPv6. psycopg2 uses libpq which handles DNS differently.
    # We'll skip the DNS check and let psycopg2 handle it directly.
    hostname = parsed.hostname
    print(f"Connecting to hostname: {hostname}")
    
    # Add connection timeout parameter
    if "connect_timeout" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        DATABASE_URL = f"{DATABASE_URL}{separator}connect_timeout=10"
    
    # Try a direct connection first to get better error messages
    # psycopg2 may handle DNS differently than Python's socket library
    try:
        print("Testing direct database connection (psycopg2 handles DNS resolution)...")
        test_conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
        with test_conn.cursor() as cur:
            cur.execute("SELECT 1")
        test_conn.close()
        print("Direct connection test successful")
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        if "could not translate host name" in error_msg.lower() or "getaddrinfo failed" in error_msg.lower():
            raise SystemExit(
                f"Connection error: {error_msg}\n\n"
                "DNS/Network issues detected. Since nslookup worked, this might be:\n"
                "1. Python's DNS resolution differs from system DNS\n"
                "2. Network/firewall blocking PostgreSQL connections\n"
                "3. IPv6 connectivity issue (Supabase uses IPv6)\n"
                "4. Corporate network restrictions\n\n"
                "Solutions to try:\n"
                "- Use a different network (mobile hotspot, home WiFi)\n"
                "- Check firewall settings (allow outbound port 5432)\n"
                "- Try using a VPN\n"
                "- Verify DATABASE_URL hostname matches Supabase Dashboard exactly\n"
                "- Contact network administrator if on corporate network"
            ) from e
        elif "password authentication failed" in error_msg.lower():
            raise SystemExit(
                f"Authentication error: {error_msg}\n\n"
                "Check your DATABASE_URL password in .env file.\n"
                "Make sure the password is correct and URL-encoded if it contains special characters."
            ) from e
        elif "tenant or user not found" in error_msg.lower() or "user not found" in error_msg.lower():
            raise SystemExit(
                f"Authentication error: {error_msg}\n\n"
                "This usually means the username in DATABASE_URL is incorrect.\n\n"
                "For Supabase pooler connections (aws-1-*.pooler.supabase.com):\n"
                "  Username format: postgres.[PROJECT_REF]\n"
                "  Example: postgres.grgosgqprembsnoftxki\n\n"
                "For direct connections (db.*.supabase.co):\n"
                "  Username: postgres\n\n"
                "To fix:\n"
                "1. Go to Supabase Dashboard > Settings > Database\n"
                "2. Check 'Connection string' section\n"
                "3. Use the 'Session mode' connection string for pooler\n"
                "4. Or use 'Transaction mode' for direct connection\n"
                "5. Copy the exact username from the connection string\n\n"
                f"Current URL (masked): {safe_url}\n"
                "Verify the username matches your Supabase project reference."
            ) from e
        else:
            raise SystemExit(f"Database connection failed: {error_msg}") from e
    
    # Create connection pool
    connection_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL
    )
    
    print("Database connection pool created successfully")
except ValueError as e:
    raise SystemExit(
        f"Invalid DATABASE_URL format: {str(e)}\n\n"
        f"Your DATABASE_URL appears to be: {DATABASE_URL[:80]}...\n\n"
        "Expected format: postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres\n\n"
        "Make sure:\n"
        "1. It starts with 'postgresql://' or 'postgres://'\n"
        "2. Format is: postgresql://username:password@hostname:port/database\n"
        "3. Password is URL-encoded if it contains special characters\n"
        "4. Get the correct URL from: Supabase Dashboard > Settings > Database > Connection string (URI)"
    ) from e
except Exception as e:
    error_msg = str(e)
    if "port" in error_msg.lower():
        raise SystemExit(
            f"Database connection error: {error_msg}\n\n"
            "This usually means your DATABASE_URL format is incorrect.\n\n"
            "Expected format: postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres\n\n"
            f"Your URL (masked): {DATABASE_URL.split('@')[0] if '@' in DATABASE_URL else DATABASE_URL[:50]}...\n\n"
            "Please check your .env file and ensure DATABASE_URL is correctly formatted.\n"
            "Get it from: Supabase Dashboard > Settings > Database > Connection string (URI)"
        ) from e
    raise SystemExit(f"Failed to create database connection pool: {str(e)}") from e

@contextmanager
def get_db_connection():
    """Get a database connection from the pool."""
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    finally:
        if conn:
            connection_pool.putconn(conn)
app = FastAPI(title="RAG API (Supabase + Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=False,  # Set to False when using "*" origins
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    k: int = 4
    session_id: str | None = None

def get_embedding(text: str):
    """Generate embeddings using Nomic model (transformers)."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        with torch.no_grad():
            outputs = embed_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return emb
    except Exception as e:
        raise RuntimeError(f"Nomic embedding failed: {e}")

def generate_from_groq(prompt: str, max_tokens: int = 500):
    """Generate responses using Groq LLMs."""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are Sabari's portfolio assistant. Provide accurate, helpful answers based on the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Slight temperature for more natural responses
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "model_not_found" in error_msg or "404" in error_msg:
            raise RuntimeError(
                f"Groq model '{GROQ_MODEL}' not found or not available.\n\n"
                f"Please check available models at https://console.groq.com/docs/models\n"
                f"Common models: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768\n"
                f"Update GROQ_MODEL in your .env file with a valid model name."
            ) from e
        raise RuntimeError(f"Groq generation failed: {e}")

def vector_to_literal(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def pg_vector_search(query_vec, k=4):
    """Perform vector search in PostgreSQL using pgvector."""
    vec_lit = vector_to_literal(query_vec)
    # Use parameterized query for safety
    sql = """
    SELECT chunk_id, document_title, chunk_text, meta,
      1 - (embedding <#> %s::vector) AS score
    FROM public.chunks
    ORDER BY embedding <#> %s::vector
    LIMIT %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (vec_lit, vec_lit, k))
                rows = cur.fetchall()
        return rows
    except psycopg2.Error as e:
        raise RuntimeError(f"Database query failed: {str(e)}") from e

@app.get("/test-groq")
def test_groq():
    """Test Groq generation endpoint for debugging."""
    import time
    test_prompt = "Say hello in one sentence."
    start = time.time()
    try:
        result = generate_from_groq(test_prompt, max_tokens=50)
        elapsed = time.time() - start
        return {
            "success": True, 
            "response": result, 
            "response_length": len(result) if result else 0,
            "time_elapsed": f"{elapsed:.2f}s"
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False, 
            "error": str(e), 
            "time_elapsed": f"{elapsed:.2f}s",
            "traceback": str(e.__traceback__)
        }

@app.get("/test-embedding")
def test_embedding():
    """Test Nomic embedding endpoint for debugging."""
    import time
    test_text = "Hello world"
    start = time.time()
    try:
        result = get_embedding(test_text)
        elapsed = time.time() - start
        return {
            "success": True,
            "embedding_length": len(result) if result else 0,
            "time_elapsed": f"{elapsed:.2f}s"
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "error": str(e),
            "time_elapsed": f"{elapsed:.2f}s"
        }

@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics."""
    return {
        "cache_size": len(_answer_cache),
        "cache_max_size": _cache_max_size,
        "cache_keys": list(_answer_cache.keys())[:10]  # Show first 10 keys
    }

@app.get("/cache/clear")
def clear_cache():
    """Clear the answer cache."""
    global _answer_cache
    size = len(_answer_cache)
    _answer_cache = {}
    return {"message": f"Cache cleared. Removed {size} entries."}

@app.get("/health")
def health():
    """Health check endpoint - fast and simple."""
    # Just return OK - don't check connections (they're checked on startup)
    # This makes health checks instant for load balancers/monitoring
    return {
        "status": "ok",
        "message": "Server is running"
    }

@app.get("/health/detailed")
def health_detailed():
    """Detailed health check with connection tests."""
    import time
    start = time.time()
    
    # Quick database ping
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Quick Groq check (test API key is set)
    groq_status = "ok"
    if not GROQ_API_KEY:
        groq_status = "error: GROQ_API_KEY not set"
    
    # Check embedding service
    embed_status = "ok (Nomic local model)"
    
    elapsed = time.time() - start
    
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "groq_llm": groq_status,
        "embeddings": embed_status,
        "embedding_dimension": VECTOR_DIM,
        "embedding_model": EMBED_MODEL,
        "response_time_ms": round(elapsed * 1000, 2)
    }

# Simple in-memory cache for answers (key: query hash, value: cached response)
_answer_cache = {}
_cache_max_size = 100  # Limit cache size

def _get_cache_key(query: str, k: int) -> str:
    """Generate cache key from query and k parameter."""
    cache_string = f"{query.lower().strip()}:{k}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def _get_cached_answer(cache_key: str):
    """Get cached answer if available."""
    return _answer_cache.get(cache_key)

def _cache_answer(cache_key: str, answer: str, sources: list, timings: dict):
    """Cache answer with LRU eviction if cache is full."""
    if len(_answer_cache) >= _cache_max_size:
        # Remove oldest entry (simple FIFO, could use LRU if needed)
        oldest_key = next(iter(_answer_cache))
        del _answer_cache[oldest_key]
    _answer_cache[cache_key] = {
        "answer": answer,
        "sources": sources,
        "timings": timings,
        "cached": True
    }

@app.post("/api/rag/query")
def rag_query(req: QueryRequest):
    import logging
    import time
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query required")
    
    # Check cache first
    cache_key = _get_cache_key(q, req.k)
    cached = _get_cached_answer(cache_key)
    if cached:
        logger.info(f"Cache HIT for query: {q}")
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "timings": {**cached["timings"], "cached": True}
        }
    
    logger.info(f"Cache MISS - Processing query: {q}")
    
    # Phase 1: Embedding
    t0 = time.time()
    try:
        q_emb = get_embedding(q)
        t_embed = time.time() - t0
        logger.info(f"Embedding completed: dimension={len(q_emb) if q_emb else 0}")
    except requests.exceptions.Timeout as e:
        t_embed = time.time() - t0
        logger.error(f"Embedding timeout after {t_embed:.2f}s: {e}")
        raise HTTPException(status_code=504, detail=f"Embedding request timed out after 30 seconds. Please try again.") from e
    except Exception as e:
        t_embed = time.time() - t0
        logger.error(f"Embedding failed after {t_embed:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}") from e
    
    # Phase 2: Vector Search
    t0 = time.time()
    try:
        results = pg_vector_search(q_emb, k=req.k)
        t_search = time.time() - t0
        logger.info(f"Vector search completed: found {len(results)} results")
    except Exception as e:
        t_search = time.time() - t0
        logger.error(f"Vector search failed after {t_search:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"vector search failed: {e}") from e
    
    if not results:
        logger.warning("No results found from vector search")
        return {"answer": "I don't have relevant information for that query.", "sources": []}
    
    # Prepare context (not timed as it's fast)
    context_parts = []
    sources = []
    for r in results:
        txt = r.get("chunk_text","")
        truncated = txt if len(txt)<1200 else txt[:1200]+"..."
        title = r.get("document_title") or (r.get("meta") or {}).get("title") or "Source"
        context_parts.append(f"[{title}] {truncated}")
        sources.append({"chunk_id": r.get("chunk_id"), "chunk_text": truncated, "score": float(r.get("score",0)), "meta": r.get("meta") or {}})
    
    # Limit context to avoid prompt being too long (reduced for speed)
    max_context_length = 1500
    truncated_context = []
    current_length = 0
    for ctx in context_parts:
        if current_length + len(ctx) > max_context_length:
            break
        truncated_context.append(ctx)
        current_length += len(ctx)
    
    prompt = ("CONTEXT:\n" + "\n\n".join(truncated_context) +
              f"\n\nUSER QUESTION: {q}\n\nINSTRUCTIONS: Based on the context above, provide a helpful and accurate answer. "
              "If the information is not in the context, say 'I don't have that information in my knowledge base.' "
              "Keep your answer concise (3-6 sentences) and mention relevant source titles when possible.")
    
    # Phase 3: LLM Generation
    t0 = time.time()
    try:
        answer = generate_from_groq(prompt, max_tokens=500)
        t_llm = time.time() - t0
        logger.info(f"LLM generation completed: response_length={len(answer) if answer else 0}")
        if not answer or not answer.strip():
            logger.warning("Groq returned empty answer, using fallback")
            answer = f"Based on the provided context, here's what I found about '{q}': " + ". ".join([f"{s.get('chunk_text', '')[:200]}..." for s in sources[:2]])
    except requests.exceptions.Timeout as e:
        t_llm = time.time() - t0
        logger.error(f"LLM generation timeout after {t_llm:.2f}s: {e}")
        raise HTTPException(status_code=504, detail=f"LLM generation timed out after 120 seconds. The request is taking longer than expected. Please try a simpler query or try again later.") from e
    except Exception as e:
        t_llm = time.time() - t0
        logger.error(f"LLM generation failed after {t_llm:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}") from e
    
    # Print timing summary
    total_time = t_embed + t_search + t_llm
    print(f"timings embed={t_embed:.2f}s search={t_search:.2f}s llm={t_llm:.2f}s total={total_time:.2f}s")
    logger.info(f"TIMING SUMMARY: embed={t_embed:.2f}s search={t_search:.2f}s llm={t_llm:.2f}s total={total_time:.2f}s")
    
    timings = {
        "embed_seconds": round(t_embed, 2),
        "search_seconds": round(t_search, 2),
        "llm_seconds": round(t_llm, 2),
        "total_seconds": round(total_time, 2),
        "cached": False
    }
    
    # Cache the answer
    _cache_answer(cache_key, answer, sources, timings)
    logger.info(f"Cached answer for query: {q}")
    
    return {
        "answer": answer, 
        "sources": sources,
        "timings": timings
    }

# Cleanup on shutdown
atexit.register(lambda: connection_pool.closeall() if connection_pool else None)
