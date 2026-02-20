from typing import List, Tuple

from search import HybridSearchSystem
from retriever import DenseRetriever
from helpers.config import EmbedderConfig

class MultiStageHybridSearch(HybridSearchSystem):
    """Multi-stage hybrid search with progressive refinement"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1_k = 50  # Initial broad retrieval
        self.stage2_k = 20  # Refined retrieval
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Multi-stage search with progressive refinement"""
        
        # Stage 1: Fast, broad retrieval
        sparse_candidates = self.sparse_retriever.search(query, self.stage1_k)
        candidate_indices = [idx for idx, _ in sparse_candidates]
        
        # Stage 2: Dense re-ranking of candidates
        if len(candidate_indices) > 0:
            candidate_docs = [
                self.dense_retriever.documents[idx] 
                for idx in candidate_indices
            ]
            
            # Create temporary dense retriever for candidates
            temp_retriever = DenseRetriever(**EmbedderConfig(
                model_name=self.dense_retriever.model.model_name,
                embedding_module=self.dense_retriever.model._embedding_module
            ))
            temp_retriever.encode_documents(candidate_docs)
            
            # Dense search within candidates
            dense_results = temp_retriever.search(query, self.stage2_k)
            
            # Map back to original indices
            final_results = [
                (candidate_indices[idx], score) 
                for idx, score in dense_results
            ]
        else:
            final_results = []
        
        return final_results[:top_k]


if __name__ == "__main__":
    from ._samples import documents
    from helpers.config import EmbedderConfig
    from helpers.print import print_query_results
    from os import getenv

    multi_stage_search = MultiStageHybridSearch(embedder_config=EmbedderConfig(
        model_name=getenv("EMBEDDING_MODEL"),
        embedding_module='local-dmr'
    ))
    multi_stage_search.index_documents(documents)

    test_queries = [
        "What are the requirements for machine learning algorithms?",
        "How do deep learning models work?",
        "What is natural language processing?"
    ]
    for query in test_queries:
        results = multi_stage_search.search(query, top_k=5)
        print_query_results(query, results, multi_stage_search.sparse_retriever.documents)