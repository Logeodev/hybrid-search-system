from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from numpy import ndarray

class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 retriever with tuning parameters.
        k1 controls term frequency saturation, while b controls length normalization.
        Use **BM25Config() from helpers.config to easily create the config dict."""
        self.k1 = k1  # Term frequency saturation point
        self.b = b    # Length normalization factor
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            token_pattern=r'\b\w+\b'
        )
        self.documents = []
        self.doc_vectors : ndarray[ndarray] = None
        self.avg_doc_length = 0
    
    def fit_documents(self, documents: List[str]):
        """Prepare BM25 index from document collection"""
        self.documents = documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        # Calculate average document length
        doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Retrieve documents using BM25 scoring"""
        query_vector = self.vectorizer.transform([query])
        
        # Calculate BM25 scores
        scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            score = self._calculate_bm25_score(query_vector, doc_vector, i)
            scores.append((i, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _calculate_bm25_score(self, query_vector, doc_vector, doc_idx: int) -> float:
        """Calculate BM25 score for query-document pair"""
        doc_length = len(self.documents[doc_idx].split())
        length_norm = self.k1 * ((1 - self.b) + self.b * (doc_length / self.avg_doc_length))
        
        # Simplified BM25 calculation using TF-IDF vectors
        tf_scores = doc_vector.toarray()[0]
        query_terms = query_vector.toarray()[0]
        
        bm25_score = 0
        for term_idx, query_weight in enumerate(query_terms):
            if query_weight > 0:
                tf = tf_scores[term_idx]
                bm25_score += query_weight * (tf * (self.k1 + 1)) / (tf + length_norm)
        
        return bm25_score
