from typing import List, Tuple, Optional
from os import getenv
import default_env

from retriever import DenseRetriever, BM25Retriever, BaseDenseRetriever, BaseBM25Retriever
from score import ScoreFusion
from helpers.config import HybridSearchConfig, EmbedderConfig, BM25Config

class HybridSearchSystem:
    def __init__(
        self, 
        dense_retriever: Optional[BaseDenseRetriever] = None,
        sparse_retriever: Optional[BaseBM25Retriever] = None,
        config: HybridSearchConfig = None,
        embedder_config: EmbedderConfig = None,
        bm25_config: BM25Config = None
    ):
        """
        Initialize hybrid search system
        
        Args:
            dense_retriever: Pre-configured dense retriever instance
            sparse_retriever: Pre-configured sparse retriever instance
            config: Configuration for hybrid search (fusion method, weights, etc.)
            embedder_config: Configuration for dense retriever's embedding model
            bm25_config: Configuration for BM25 retriever parameters
        > The configurations objects will be ignored if the corresponding retriever instances are provided.
        """
        if config is None:
            config = HybridSearchConfig()
        if embedder_config is None:
            embedder_config = EmbedderConfig()
        if bm25_config is None:
            bm25_config = BM25Config()
        self.dense_retriever = dense_retriever or DenseRetriever(**embedder_config)
        self.sparse_retriever = sparse_retriever or BM25Retriever(**bm25_config)
        self.fusion_method = config.fusion_method
        self.dense_weight = config.dense_weight
        self.sparse_weight = config.sparse_weight
        self.score_fusion = ScoreFusion()
        self.documents: List[str] = []
        
    def index_documents(self, documents: List[str]):
        """Index documents for both dense and sparse retrieval"""
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents
        
        # Index for dense retrieval
        self.dense_retriever.encode_documents(documents)
        
        # Index for sparse retrieval
        self.sparse_retriever.fit_documents(documents)
        
        print("Indexing complete!")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (document_index, combined_score) tuples
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, top_k * 2)
        sparse_results = self.sparse_retriever.search(query, top_k * 2)
        
        # Combine results using specified fusion method
        if self.fusion_method == "rrf":
            combined_results = self.score_fusion.reciprocal_rank_fusion(
                [dense_results, sparse_results]
            )
        elif self.fusion_method == "weighted_sum":
            combined_results = self.score_fusion.weighted_sum_fusion(
                dense_results, 
                sparse_results,
                self.dense_weight,
                self.sparse_weight
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return combined_results[:top_k]
    
    def get_documents_by_indices(self, indices: List[int]) -> List[str]:
        """Retrieve document texts by their indices"""
        return [self.documents[i] for i in indices]
    

if __name__ == "__main__":
    # Sample documents
    from ._samples import documents
    
    embed_conf = EmbedderConfig(
        model_name=getenv("EMBEDDING_MODEL"), embedding_module='local-dmr')
    search_conf = HybridSearchConfig(dense_weight=0.6, sparse_weight=0.4)
    # Initialize hybrid search
    hybrid_search = HybridSearchSystem(
        config=search_conf,
        embedder_config=embed_conf
    )
    
    # Index documents
    hybrid_search.index_documents(documents)
    
    # Perform search
    query = "neural networks for machine learning"
    results = hybrid_search.search(query, top_k=5)
    
    print(f"Query: {query}")
    print("\nHybrid Search Results:")
    for rank, (doc_idx, score) in enumerate(results, 1):
        doc_text = documents[doc_idx]
        print(f"{rank}. Score: {score:.4f}")
        print(f"   Document: {doc_text}\n")
