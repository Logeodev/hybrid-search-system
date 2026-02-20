import numpy as np
from typing import List, Set

class RetrievalMetrics:
    """Calculate retrieval quality metrics"""
    
    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate precision at k"""
        retrieved_k = set(retrieved[:k])
        return len(retrieved_k & relevant) / min(k, len(retrieved_k))
    
    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate recall at k"""
        retrieved_k = set(retrieved[:k])
        return len(retrieved_k & relevant) / len(relevant) if relevant else 0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_lists: List[List[int]], relevant_lists: List[Set[int]]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            rr = 0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    rr = 1 / rank
                    break
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def normalized_dcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k"""
        def dcg_at_k(relevance_scores: List[int], k: int) -> float:
            dcg = 0
            for i, score in enumerate(relevance_scores[:k]):
                dcg += score / np.log2(i + 2)
            return dcg
        
        # Binary relevance scores
        relevance_scores = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0