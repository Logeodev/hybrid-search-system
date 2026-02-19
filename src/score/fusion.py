from typing import Dict, List, Tuple
import numpy as np

class ScoreFusion:
    """Combine scores from multiple retrieval methods"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[List[Tuple[int, float]]], 
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine ranked lists using Reciprocal Rank Fusion
        RRF score = sum(1 / (rank + k)) for each retrieval method
        """
        doc_scores = {}
        
        for results in results_list:
            for rank, (doc_id, score) in enumerate(results):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1 / (rank + k)
        
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(doc_id, score) for doc_id, score in sorted_results]
    
    @staticmethod
    def weighted_sum_fusion(
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Combine scores using weighted sum
        Final score = (dense_weight * dense_score) + (sparse_weight * sparse_score)
        """
        # Normalize scores to [0, 1] range
        dense_scores = ScoreFusion._normalize_scores(dense_results)
        sparse_scores = ScoreFusion._normalize_scores(sparse_results)
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for doc_id, score in dense_scores.items():
            combined_scores[doc_id] = dense_weight * score
        
        # Add sparse scores
        for doc_id, score in sparse_scores.items():
            if doc_id in combined_scores:
                combined_scores[doc_id] += sparse_weight * score
            else:
                combined_scores[doc_id] = sparse_weight * score
        
        # Sort and return
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(doc_id, score) for doc_id, score in sorted_results]
    
    @staticmethod
    def _normalize_scores(results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range"""
        if not results:
            return {}
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id, _ in results}
        
        normalized = {}
        for doc_id, score in results:
            normalized[doc_id] = (score - min_score) / (max_score - min_score)
        
        return normalized
