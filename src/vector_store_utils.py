"""
Vector Store Utilities for Complaint Analysis RAG System

This module provides utilities for loading and querying the complaint vector store
created in Task 2. It includes the ComplaintVectorStore class for easy interaction
with the FAISS index and associated metadata. 
"""

import os
import json
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class ComplaintVectorStore:
    """
    Utility class for loading and querying the complaint vector store.
    
    This class provides a convenient interface for:
    - Loading pre-built FAISS index and associated data
    - Performing semantic search queries
    - Filtering results by product category
    - Retrieving statistics about the vector store
    """
    
    def __init__(self, vector_store_dir: str):
        """
        Initialize the ComplaintVectorStore.
        
        Args:
            vector_store_dir: Path to directory containing vector store files
        """
        self.vector_store_dir = vector_store_dir
        self.index = None
        self.chunks = None
        self.metadata = None
        self.config = None
        self.embedding_model = None
        
    def load(self) -> None:
        """
        Load all vector store components from disk.
        
        This method loads:
        - FAISS index for similarity search
        - Text chunks
        - Metadata for each chunk
        - Configuration parameters
        - Embedding model
        """
        print("Loading vector store components...")
        
        # Load configuration
        config_path = os.path.join(self.vector_store_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load FAISS index
        faiss_path = os.path.join(self.vector_store_dir, 'complaint_index.faiss')
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            
        self.index = faiss.read_index(faiss_path)
        
        # Load chunks
        chunks_path = os.path.join(self.vector_store_dir, 'chunks.pkl')
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
            
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(self.vector_store_dir, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(self.config['embedding_model'])
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
        
        # Validate data consistency
        if len(self.chunks) != len(self.metadata):
            raise ValueError("Mismatch between number of chunks and metadata entries")
            
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Mismatch between FAISS index size and number of chunks")
        
        print(f"Vector store loaded successfully!")
        print(f"- {len(self.chunks)} chunks")
        print(f"- {self.index.ntotal} vectors in index")
        print(f"- Embedding model: {self.config['embedding_model']}")
    
    def search(self, query: str, k: int = 5, filter_category: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_category: Optional category filter (e.g., 'Credit Cards')
            
        Returns:
            List of dictionaries containing search results with text, score, and metadata
        """
        if self.index is None:
            raise ValueError("Vector store not loaded. Call load() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search with extra results to allow for filtering
        search_k = k * 3 if filter_category else k
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if len(results) >= k:
                break
                
            # Skip invalid indices
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            metadata = self.metadata[idx]
            
            # Apply category filter if specified
            if filter_category and metadata.get('category') != filter_category:
                continue
            
            results.append({
                'text': self.chunks[idx],
                'score': float(score),
                'metadata': metadata,
                'chunk_id': int(idx)
            })
        
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: Index of the chunk to retrieve
            
        Returns:
            Dictionary containing chunk text and metadata
        """
        if self.chunks is None:
            raise ValueError("Vector store not loaded. Call load() first.")
            
        if chunk_id < 0 or chunk_id >= len(self.chunks):
            raise IndexError(f"Chunk ID {chunk_id} out of range")
        
        return {
            'text': self.chunks[chunk_id],
            'metadata': self.metadata[chunk_id],
            'chunk_id': chunk_id
        }
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics about the vector store.
        
        Returns:
            Dictionary containing various statistics and metadata
        """
        if self.chunks is None:
            raise ValueError("Vector store not loaded. Call load() first.")
        
        # Category distribution
        categories = [meta.get('category', 'Unknown') for meta in self.metadata]
        category_counts = pd.Series(categories).value_counts().to_dict()
        
        # Product distribution
        products = [meta.get('product', 'Unknown') for meta in self.metadata]
        product_counts = pd.Series(products).value_counts().to_dict()
        
        # Issue distribution (top 10)
        issues = [meta.get('issue', 'Unknown') for meta in self.metadata]
        issue_counts = pd.Series(issues).value_counts().head(10).to_dict()
        
        # Chunk statistics
        chunk_lengths = [len(chunk) for chunk in self.chunks]
        
        return {
            'total_chunks': len(self.chunks),
            'total_complaints': len(set(meta.get('complaint_id') for meta in self.metadata if meta.get('complaint_id'))),
            'category_distribution': category_counts,
            'product_distribution': product_counts,
            'top_issues': issue_counts,
            'chunk_length_stats': {
                'mean': np.mean(chunk_lengths),
                'median': np.median(chunk_lengths),
                'min': np.min(chunk_lengths),
                'max': np.max(chunk_lengths),
                'std': np.std(chunk_lengths)
            },
            'embedding_dim': self.index.d if self.index else 0,
            'config': self.config
        }
    
    def search_by_category(self, category: str, query: str = "", k: int = 10) -> List[Dict]:
        """
        Search within a specific product category.
        
        Args:
            category: Product category to search within
            query: Optional search query (if empty, returns random samples)
            k: Number of results to return
            
        Returns:
            List of search results from the specified category
        """
        if query:
            return self.search(query, k=k, filter_category=category)
        else:
            # Return random samples from the category
            category_indices = [
                i for i, meta in enumerate(self.metadata) 
                if meta.get('category') == category
            ]
            
            if not category_indices:
                return []
            
            # Sample random indices
            sample_size = min(k, len(category_indices))
            sampled_indices = np.random.choice(category_indices, sample_size, replace=False)
            
            results = []
            for idx in sampled_indices:
                results.append({
                    'text': self.chunks[idx],
                    'score': 1.0,  # No similarity score for random sampling
                    'metadata': self.metadata[idx],
                    'chunk_id': int(idx)
                })
            
            return results
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available product categories.
        
        Returns:
            List of unique category names
        """
        if self.metadata is None:
            raise ValueError("Vector store not loaded. Call load() first.")
        
        categories = set(meta.get('category', 'Unknown') for meta in self.metadata)
        return sorted(list(categories))
    
    def export_search_results(self, results: List[Dict], filename: str) -> None:
        """
        Export search results to a CSV file.
        
        Args:
            results: Search results from search() method
            filename: Output filename
        """
        if not results:
            print("No results to export")
            return
        
        # Flatten results for CSV export
        export_data = []
        for result in results:
            row = {
                'text': result['text'],
                'score': result['score'],
                'chunk_id': result.get('chunk_id', -1)
            }
            # Add metadata fields
            for key, value in result['metadata'].items():
                row[f'meta_{key}'] = value
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def load_vector_store(vector_store_dir: str = '../vector_store') -> ComplaintVectorStore:
    """
    Convenience function to load and return a ComplaintVectorStore instance.
    
    Args:
        vector_store_dir: Path to vector store directory
        
    Returns:
        Loaded ComplaintVectorStore instance
    """
    store = ComplaintVectorStore(vector_store_dir)
    store.load()
    return store


if __name__ == "__main__":
    # Example usage
    print("Testing ComplaintVectorStore...")
    
    try:
        # Load vector store
        store = load_vector_store('../vector_store')
        
        # Get statistics
        stats = store.get_stats()
        print(f"\nVector Store Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total complaints: {stats['total_complaints']}")
        
        # Test search
        results = store.search("credit card billing problem", k=3)
        print(f"\nSearch results for 'credit card billing problem':")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Category: {result['metadata']['category']}")
            print(f"   Text: {result['text'][:100]}...")
        
        print("\nComplaintVectorStore test completed successfully!")
        
    except Exception as e:
        print(f"Error testing ComplaintVectorStore: {e}")