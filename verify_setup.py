import os
import sys

def verify_setup():
    print("🧪 Verifying setup...")
    
    # 1. Check Imports
    try:
        from sentence_transformers import SentenceTransformer
        print(f"✅ sentence-transformers installed: {SentenceTransformer.__module__}")
    except ImportError:
        print("❌ sentence-transformers NOT installed. Run: pip install -r requirements.txt")
        return

    # 2. Check Embedding Model Loading
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"🔄 Loading model {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Dimensions: {dim}")
        
        if dim != 384:
            print(f"⚠️ WARNING: Expected 384 dimensions, got {dim}. Check your DB column!")
        else:
            print("✅ Dimensions match MiniLM standard (384).")
            
        # 3. Test Encoding
        vec = model.encode("hello world", normalize_embeddings=True)
        print(f"✅ Test encoding successful. Vector length: {len(vec)}")
        
    except Exception as e:
        print(f"❌ Failed to load model or encode: {e}")
        return

    print("\n🎉 Setup verification complete. You can now restart your backend.")

if __name__ == "__main__":
    verify_setup()
