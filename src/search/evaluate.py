import numpy as np
from typing import List, Set, Dict

from .hybrid_rag import HybridSearchSystem
from score import RetrievalMetrics

def evaluate_search_system(
    search_system: HybridSearchSystem,
    test_queries: List[str],
    ground_truth: List[Set[int]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Comprehensive evaluation of search system"""
    
    all_retrieved = []
    metrics = RetrievalMetrics()
    
    # Collect results for all queries
    for query in test_queries:
        results = search_system.search(query, top_k=max(k_values))
        retrieved_indices = [doc_idx for doc_idx, _ in results]
        all_retrieved.append(retrieved_indices)
    
    # Calculate metrics
    evaluation_results = {}
    
    for k in k_values:
        precisions = [
            metrics.precision_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(all_retrieved, ground_truth)
        ]
        recalls = [
            metrics.recall_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(all_retrieved, ground_truth)
        ]
        ndcgs = [
            metrics.normalized_dcg_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(all_retrieved, ground_truth)
        ]
        
        evaluation_results.update({
            f'precision@{k}': np.mean(precisions),
            f'recall@{k}': np.mean(recalls),
            f'ndcg@{k}': np.mean(ndcgs)
        })
    
    # Calculate MRR
    evaluation_results['mrr'] = metrics.mean_reciprocal_rank(all_retrieved, ground_truth)
    
    return evaluation_results

if __name__ == "__main__":
    from ._samples import documents, queries as test_queries, ground_truth
    from helpers.config import EmbedderConfig, HybridSearchConfig
    from os import getenv
    from .staged_hybrid_rag import MultiStageHybridSearch
    from .optimize import optimize_fusion_weights
    from tabulate import tabulate

    embed_conf = EmbedderConfig(
        model_name=getenv("EMBEDDING_MODEL"), embedding_module='local-dmr')
    base_conf = HybridSearchConfig(dense_weight=0.5, sparse_weight=0.5, fusion_method="weighted_sum")
    base_system = HybridSearchSystem(config=base_conf, embedder_config=embed_conf)
    base_system.index_documents(documents)

    # Evaluate base system
    results_base = evaluate_search_system(base_system, test_queries, ground_truth)

    # Optimize fusion weights using the provided function
    print("\nOptimizing fusion weights...")
    best_weights = optimize_fusion_weights(base_system, test_queries, ground_truth)
    print(f"Best fusion weights found: {best_weights}")

    # Create and evaluate MultiStageHybridSearch with optimized weights
    opti_config = HybridSearchConfig(dense_weight=best_weights[0], sparse_weight=best_weights[1])
    optimized_system = HybridSearchSystem(embedder_config=embed_conf, config=opti_config)
    optimized_system.index_documents(documents)
    results_optimized = evaluate_search_system(optimized_system, test_queries, ground_truth)

    # Compare to multistage system
    multistage_system = MultiStageHybridSearch(embedder_config=embed_conf, config=opti_config)
    multistage_system.index_documents(documents)
    results_multistage = evaluate_search_system(multistage_system, test_queries, ground_truth)
    
    # Results
    # Prepare data for table
    metrics = list(results_base.keys())
    table = [
        ["Metric", "Base Hybrid", "Optimized Hybrid", "MultiStage Hybrid"]
    ]
    for metric in metrics:
        row = [
            metric,
            f"{results_base[metric]:.4f}",
            f"{results_optimized[metric]:.4f}",
            f"{results_multistage[metric]:.4f}"
        ]
        table.append(row)

    print("\nEvaluation Results Comparison:\n")
    print(tabulate(table[1:], headers=table[0], tablefmt="github"))