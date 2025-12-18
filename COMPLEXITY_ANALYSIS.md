# Time and Space Complexity Analysis

## Overview
This RAG (Retrieval-Augmented Generation) system uses:
- **Embeddings**: Nomic AI model (768 dimensions)
- **Vector Search**: PostgreSQL with pgvector (IVFFlat index)
- **LLM**: Groq API (llama-3.1-70b-instant)
- **Caching**: In-memory hash table

---

## Time Complexity

### 1. Embedding Generation (`get_embedding`)
**Function**: `get_embedding(text: str)`

- **Tokenization**: O(n) where n = input text length
- **Model Forward Pass**: O(n × d) where d = embedding dimension (768)
- **Mean Pooling**: O(n × d)
- **Total**: **O(n × d)** = **O(768n)** ≈ **O(n)**

**Practical**: ~10-50ms for typical queries (50-200 tokens)

---

### 2. Vector Search (`pg_vector_search`)
**Function**: `pg_vector_search(query_vec, k=4)`

- **Index Lookup (IVFFlat)**: O(log N) where N = total chunks in database
- **Distance Calculation**: O(k × d) where k = number of results, d = 768
- **Sorting**: O(k log k) for top-k results
- **Total**: **O(log N + k × d + k log k)**

**Practical**: 
- With IVFFlat index: ~5-20ms for N=10,000 chunks
- Without index: O(N × d) = O(768N) - much slower!

---

### 3. LLM Generation (`generate_from_groq`)
**Function**: `generate_from_groq(prompt, max_tokens=500)`

- **API Call**: O(1) network overhead
- **Generation**: O(t) where t = number of tokens generated
- **Total**: **O(t)** where t ≤ max_tokens

**Practical**: ~200-1000ms depending on response length (Groq is fast!)

---

### 4. Cache Operations
**Functions**: `_get_cache_key`, `_get_cached_answer`, `_cache_answer`

- **Hash Key Generation**: O(n) where n = query length
- **Cache Lookup**: O(1) - hash table lookup
- **Cache Insert**: O(1) - hash table insert
- **Cache Eviction**: O(1) - FIFO removal
- **Total**: **O(1)** for cache operations

---

### 5. Complete RAG Query Pipeline (`rag_query`)
**Function**: `@app.post("/api/rag/query")`

**Breakdown**:
1. **Cache Check**: O(1)
2. **Embedding**: O(n × d) where n = query length
3. **Vector Search**: O(log N + k × d)
4. **Context Preparation**: O(k × c) where c = average chunk size
5. **LLM Generation**: O(t) where t = response tokens

**Total Time Complexity**: 
```
O(1) + O(n×d) + O(log N + k×d) + O(k×c) + O(t)
= O(n×d + log N + k×d + k×c + t)
```

**Simplified** (assuming constants):
- **Best Case** (cache hit): **O(1)**
- **Worst Case** (no cache): **O(n + log N + k + t)**

**Practical Performance**:
- Cache hit: ~1-5ms
- Cache miss: ~200-1500ms total
  - Embedding: ~10-50ms
  - Vector search: ~5-20ms
  - LLM generation: ~200-1000ms

---

## Space Complexity

### 1. Embedding Model (Startup)
- **Model Weights**: ~500MB - 1GB (Nomic model)
- **Tokenizer**: ~10-50MB
- **Total**: **O(1)** - constant, loaded once at startup

---

### 2. Embedding Vectors
- **Single Embedding**: O(d) = O(768) = **O(1)** constant
- **Query Embedding**: 768 floats × 4 bytes = **3KB**
- **All Chunks in DB**: O(N × d) where N = number of chunks
  - Example: 10,000 chunks × 768 × 4 bytes = **~30MB**

---

### 3. Vector Search
- **Query Vector**: O(d) = **O(1)**
- **Result Set**: O(k × chunk_size) where k = results returned
  - Example: k=4, avg chunk=500 chars = **~2KB**

---

### 4. LLM Context
- **Prompt Construction**: O(k × c) where c = chunk size
  - Example: k=4, c=1200 chars = **~5KB**
- **Response**: O(t) where t = tokens generated
  - Example: 500 tokens = **~2KB**

---

### 5. Cache
- **Cache Size**: O(C × R) where:
  - C = cache_max_size (100 entries)
  - R = average response size (~5-10KB)
- **Total Cache Memory**: **O(100 × 10KB) = ~1MB**

---

### 6. Database Connection Pool
- **Pool Size**: 10 connections
- **Per Connection**: ~1-5MB
- **Total**: **~10-50MB**

---

## Overall Space Complexity

**Runtime Memory**:
```
O(1) [model] + O(N×d) [DB embeddings] + O(C×R) [cache] + O(pool)
= O(N×d + C×R + pool)
```

**Practical**:
- **Startup**: ~500MB-1GB (model loading)
- **Runtime**: ~50-100MB (excluding model)
- **Database**: ~30MB per 10,000 chunks

---

## Scalability Analysis

### Current Limits
- **Chunks**: ~100,000 chunks (before performance degrades)
- **Concurrent Users**: ~10-50 (connection pool limited)
- **Cache Size**: 100 entries (configurable)

### Bottlenecks
1. **Embedding Generation**: CPU-bound, ~10-50ms per query
2. **Vector Search**: Database-bound, scales with O(log N)
3. **LLM Generation**: Network-bound (Groq API), ~200-1000ms

### Optimization Opportunities

1. **Batch Embeddings**: 
   - Current: O(n) per query
   - Optimized: O(n/batch_size) with batching
   - **Improvement**: 5-10x faster for multiple queries

2. **Vector Index Tuning**:
   - Current: IVFFlat with default lists
   - Optimized: Tune `lists` parameter for your data size
   - **Improvement**: 2-5x faster searches

3. **Cache Strategy**:
   - Current: FIFO eviction
   - Optimized: LRU eviction
   - **Improvement**: Better cache hit rate

4. **Connection Pooling**:
   - Current: 10 connections
   - Optimized: Scale based on load
   - **Improvement**: Better concurrency

---

## Summary Table

| Component | Time Complexity | Space Complexity | Practical Time |
|-----------|----------------|------------------|----------------|
| Embedding | O(n×d) | O(d) | 10-50ms |
| Vector Search | O(log N + k×d) | O(k×c) | 5-20ms |
| LLM Generation | O(t) | O(t) | 200-1000ms |
| Cache Lookup | O(1) | O(1) | <1ms |
| **Full Query** | **O(n + log N + k + t)** | **O(k×c + t)** | **200-1500ms** |

**Where**:
- n = query text length
- d = embedding dimension (768)
- N = total chunks in database
- k = number of results (default: 4)
- c = average chunk size
- t = response tokens generated

---

## Real-World Performance

**Typical Query** ("What is your experience?"):
- Query length: ~20 tokens
- Cache miss: ~300-800ms
- Cache hit: ~1-5ms

**Heavy Query** (long question + many results):
- Query length: ~100 tokens
- k = 10 results
- Cache miss: ~800-2000ms
- Cache hit: ~1-5ms

**Throughput**:
- **Without cache**: ~1-5 queries/second
- **With cache** (50% hit rate): ~10-20 queries/second

---

## Recommendations

1. **For Scale**:
   - Use batch embedding for multiple queries
   - Tune IVFFlat index parameters
   - Implement LRU cache
   - Consider Redis for distributed caching

2. **For Performance**:
   - Pre-warm cache with common queries
   - Use connection pooling effectively
   - Monitor Groq API rate limits
   - Consider async processing for embeddings

3. **For Cost**:
   - Current: $0 (local embeddings) + Groq free tier
   - At scale: Monitor Groq API usage
   - Consider caching aggressively to reduce API calls

