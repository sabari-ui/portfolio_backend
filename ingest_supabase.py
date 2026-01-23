# ingest_supabase.py
import os
import uuid
import hashlib
import time
from dotenv import load_dotenv
from supabase import create_client
import fitz  # PyMuPDF
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

load_dotenv()

# ---------------------------
# ENV VARIABLES
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
DOC_PATH = os.getenv("DOCUMENT_PATH", "../portfolio_frontend/docs/SABARI.docx")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "6"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("❌ ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

# ---------------------------
# INIT SUPABASE CLIENT
# ---------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# LOAD NOMIC EMBEDDING MODEL
# ---------------------------
print(f"🚀 Loading Nomic embedding model: {EMBED_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
model.eval()

VECTOR_DIM = model.config.hidden_size
print(f"✅ Nomic model loaded. Embedding dimension = {VECTOR_DIM}")

def get_embedding(text: str):
    """Return a single embedding vector (768D)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return emb

def get_embeddings_batch(text_list):
    """Return batch embeddings (much faster)"""
    inputs = tokenizer(text_list, return_tensors="pt", truncation=True, padding=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
    return batch_emb

# ---------------------------
# SEMANTIC CHUNKING
# ---------------------------
def is_heading(line: str):
    text = line.strip()
    if not text:
        return False
    if text.isupper() and len(text) > 3:
        return True
    if text.endswith(":") and len(text.split()) <= 6:
        return True
    return False

def semantic_chunk(text, chunk_max=1000, overlap_words=40):
    text = text.replace("\r\n", "\n")
    blocks = []
    current = []

    for line in text.split("\n"):
        if is_heading(line):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            current.append(line)

        elif line.strip() == "":
            if current:
                blocks.append("\n".join(current).strip())
                current = []

        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    # Now break blocks into final chunks
    final_chunks = []
    for block in blocks:
        words = block.split()
        if len(block) <= chunk_max:
            final_chunks.append(block)
            continue

        start = 0
        while start < len(words):
            cur_words = words[start:start + 200]
            slice_text = " ".join(cur_words)
            final_chunks.append(slice_text)
            start += (200 - overlap_words)

    return [c for c in final_chunks if len(c.strip()) > 20]

# ---------------------------
# EXTRACT SEMANTIC CHUNKS
# ---------------------------
def extract_semantic_chunks(path):
    doc = fitz.open(path)
    chunks = []

    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if not text or not text.strip():
            continue

        sem_chunks = semantic_chunk(text)

        for idx, ch in enumerate(sem_chunks):
            chunks.append({
                "document_title": os.path.splitext(os.path.basename(path))[0],
                "chunk_text": ch,
                "meta": {"page": i + 1, "chunk_index": idx}
            })

    return chunks

def hash_chunk(text):
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

# ---------------------------
# INSERT INTO SUPABASE
# ---------------------------
def upsert_batch(rows):
    supabase.table("chunks").upsert(
        rows, on_conflict="chunk_hash"
    ).execute()
    print(f"Inserted/Updated {len(rows)} chunks")

# ---------------------------
# INGEST PIPELINE
# ---------------------------
def ingest(path):
    chunks = extract_semantic_chunks(path)
    print(f"📄 Extracted semantic chunks: {len(chunks)}")

    # Deduplicate
    final_list = []
    seen = set()

    for ch in chunks:
        h = hash_chunk(ch["chunk_text"])
        if h in seen:
            continue
        seen.add(h)

        final_list.append({
            "chunk_id": str(uuid.uuid4()),
            "chunk_hash": h,
            "document_title": ch["document_title"],
            "chunk_text": ch["chunk_text"],
            "meta": ch["meta"]
        })

    print(f"🧹 After dedupe: {len(final_list)} chunks")

    # Batch embedding + upsert
    for i in tqdm(range(0, len(final_list), BATCH_SIZE)):
        batch = final_list[i:i + BATCH_SIZE]
        texts = [b["chunk_text"] for b in batch]

        t0 = time.time()
        embeddings = get_embeddings_batch(texts)
        print(f"⚡ Embedded batch of {len(texts)} in {time.time() - t0:.2f}s")

        rows = []
        for ch, emb in zip(batch, embeddings):
            ch["embedding"] = emb
            rows.append(ch)

        upsert_batch(rows)

    print("🎉 Ingestion complete!")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    ingest(DOC_PATH)
