from retriever import DenseRetriever, BM25Retriever
from score import ScoreFusion
from typing import List, Tuple

class HybridSearchSystem:
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        fusion_method: str = "rrf",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid search system
        
        Args:
            embedding_model: Sentence transformer model name
            fusion_method: "rrf" or "weighted_sum"
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
        """
        self.dense_retriever = DenseRetriever(embedding_model)
        self.sparse_retriever = BM25Retriever()
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.score_fusion = ScoreFusion()
        
    def index_documents(self, documents: List[str]):
        """Index documents for both dense and sparse retrieval"""
        print(f"Indexing {len(documents)} documents...")
        
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
        return [self.dense_retriever.documents[i] for i in indices]

# Example usage
def demonstrate_hybrid_search():
    """Show hybrid search in action"""
    
    # Sample documents
    documents = [
        "Machine learning algorithms require large datasets for training.",
        "Deep learning models use neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision systems can identify objects in images.",
        "Reinforcement learning agents learn through trial and error.",
        "Supervised learning uses labeled data to train models.",
        "Unsupervised learning discovers patterns in unlabeled data.",
        "Transfer learning adapts pre-trained models to new tasks."
    ]
    
    # Initialize hybrid search
    hybrid_search = HybridSearchSystem(
        fusion_method="rrf",
        dense_weight=0.6,
        sparse_weight=0.4
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

if __name__ == "__main__":
    demonstrate_hybrid_search()
