from typing import List, Tuple
from numpy import arange
from os import getenv
import default_env

from search import HybridSearchSystem

def optimize_fusion_weights(
    hybrid_search: HybridSearchSystem,
    test_queries: List[str],
    ground_truth: List[List[int]],
    weight_range: Tuple[float, float] = (0.3, 0.8)
) -> Tuple[float, float]:
    """
    Find optimal fusion weights using validation data
    
    Args:
        hybrid_search: Hybrid search instance
        test_queries: List of test queries
        ground_truth: List of relevant document indices for each query
        weight_range: Range of weights to test
        
    Returns:
        Optimal (dense_weight, sparse_weight) tuple
    """
    best_score = 0
    best_weights = (0.5, 0.5)
    
    # Test different weight combinations
    for dense_weight in arange(weight_range[0], weight_range[1], 0.1):
        sparse_weight = 1.0 - dense_weight
        
        # Update weights
        hybrid_search.dense_weight = dense_weight
        hybrid_search.sparse_weight = sparse_weight
        
        # Calculate average precision
        total_precision = 0
        for query, relevant_docs in zip(test_queries, ground_truth):
            results = hybrid_search.search(query, top_k=10)
            retrieved_docs = [doc_idx for doc_idx, _ in results]
            
            # Calculate precision at k
            relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
            precision = relevant_retrieved / len(retrieved_docs)
            total_precision += precision
        
        avg_precision = total_precision / len(test_queries)
        
        if avg_precision > best_score:
            best_score = avg_precision
            best_weights = (dense_weight, sparse_weight)
    
    return best_weights

if __name__ == "__main__":
    from ._samples import documents
    from helpers.config import EmbedderConfig

    # Example usage
    hybrid_search = HybridSearchSystem(embedder_config=EmbedderConfig(
        model_name=getenv("EMBEDDING_MODEL"), 
        embedding_module='local-dmr')
    )
    hybrid_search.index_documents(documents)

    test_queries = [
        "What are the requirements for machine learning algorithms?",
        "How do deep learning models work?",
        "What is natural language processing?"
    ]
    ground_truth = [
        [0],  # Relevant document indices for query 1
        [1],  # Relevant document indices for query 2
        [2]   # Relevant document indices for query 3
    ]
    optimal_weights = optimize_fusion_weights(hybrid_search, test_queries, ground_truth)
    print(f"Optimal Weights: Dense={optimal_weights[0]:.2f}, Sparse={optimal_weights[1]:.2f}")
    