import os
import sys
from pathlib import Path

# Add current dir to path
sys.path.append(str(Path(__file__).parent))

from retrieval import Reranker

def debug():
    print("Initializing Reranker...")
    try:
        # Mock config
        os.environ["GROQ_API_KEY"] = "mock"
        
        reranker = Reranker()
        query = "What is AI?"
        # Test with various types
        docs = [
            "AI is artificial intelligence.",
            None,
            "",
            123,
            ["not a string"]
        ]
        
        print(f"Reranking documents: {docs}")
        indices, scores = reranker.rerank(query, docs)
        print(f"Success! Indices: {indices}, Scores: {scores}")
        
    except Exception as e:
        import traceback
        print(f"Caught expected/unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug()
