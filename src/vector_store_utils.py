
import os
import json
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class ComplaintVectorStore:
    """
    Utility class for loading and querying the complaint vector store.
    """

    def __init__(self, vector_store_dir: str):
        self.vector_store_dir = vector_store_dir
        self.index = None
        self.chunks = None
        self.metadata = None
        self.config = None
        self.embedding_model = None

    def load(self):
        """Load all vector store components."""
        print("Loading vector store components...")

        # Load configuration
        config_path = os.path.join(self.vector_store_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load FAISS index
        faiss_path = os.path.join(self.vector_store_dir, 'complaint_index.faiss')
        self.index = faiss.read_index(faiss_path)

        # Load chunks
        chunks_path = os.path.join(self.vector_store_dir, 'chunks.pkl')
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        # Load metadata
        metadata_path = os.path.join(self.vector_store_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.config['embedding_model'])

        print(f"Vector store loaded successfully!")
        print(f"- {len(self.chunks)} chunks")
        print(f"- {self.index.ntotal} vectors in index")
        print(f"- Embedding model: {self.config['embedding_model']}")

    def search(self, query: str, k: int = 5, filter_category: str = None) -> List[Dict]:
        """Search for similar chunks."""
        if self.index is None:
            raise ValueError("Vector store not loaded. Call load() first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k * 2)  # Get more for filtering

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if len(results) >= k:
                break

            metadata = self.metadata[idx]

            # Apply category filter if specified
            if filter_category and metadata['category'] != filter_category:
                continue

            results.append({
                'text': self.chunks[idx],
                'score': float(score),
                'metadata': metadata
            })

        return results

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        if self.chunks is None:
            raise ValueError("Vector store not loaded. Call load() first.")

        # Category distribution
        categories = [meta['category'] for meta in self.metadata]
        category_counts = pd.Series(categories).value_counts().to_dict()

        return {
            'total_chunks': len(self.chunks),
            'total_complaints': len(set(meta['complaint_id'] for meta in self.metadata)),
            'category_distribution': category_counts,
            'avg_chunk_length': np.mean([len(chunk) for chunk in self.chunks]),
            'config': self.config
        }
