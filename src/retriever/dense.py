import numpy as np
from typing import List, Tuple
from documents import Document
from .base import BaseDenseRetriever

class DenseRetriever(BaseDenseRetriever):    
    def encode_documents(self, documents: List[Document]) -> np.ndarray:
        """Convert documents to dense vectors"""
        self.documents = documents
        texts = [doc.text for doc in documents]
        self.document_embeddings = np.array(self.model.encode(texts), copy=True)
        return self.document_embeddings
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar documents using cosine similarity"""
        query_embedding = np.array(self.model.encode(query))
        
        # Calculate cosine similarity
        similarities = np.dot(query_embedding, self.document_embeddings.T)
        similarities = similarities / (
            np.linalg.norm(query_embedding) * 
            np.linalg.norm(self.document_embeddings, axis=1)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        results = [(str(idx), similarities[0][idx]) for idx in top_indices]
        
        return results
