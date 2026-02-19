from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize dense retriever with embedding model"""
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = None
        self.documents = []
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Convert documents to dense vectors"""
        self.documents = documents
        self.document_embeddings = self.model.encode(documents)
        return self.document_embeddings
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find most similar documents using cosine similarity"""
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(query_embedding, self.document_embeddings.T)
        similarities = similarities / (
            np.linalg.norm(query_embedding) * 
            np.linalg.norm(self.document_embeddings, axis=1)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        results = [(idx, similarities[0][idx]) for idx in top_indices]
        
        return results
