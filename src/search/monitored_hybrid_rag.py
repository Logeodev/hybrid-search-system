from typing import List, Tuple, Dict, Any
import time
from search import HybridSearchSystem
from score import PerformanceMonitor
import default_env

# Integrate monitoring into hybrid search
class MonitoredHybridSearch(HybridSearchSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = PerformanceMonitor()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search with performance monitoring"""
        start_time = time.time()
        
        results = super().search(query, top_k)
        
        query_time = time.time() - start_time
        self.monitor.record_query(query_time)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.monitor.get_performance_report()
    
if __name__ == "__main__":
    from ._samples import documents
    from os import getenv
    from helpers.config import EmbedderConfig
    from helpers.print import print_query_results

    # Example usage
    monitored_search = MonitoredHybridSearch(embedder_config=EmbedderConfig(
        model_name=getenv("EMBEDDING_MODEL"), 
        embedding_module='local-dmr'
        ))
    monitored_search.index_documents(documents)

    test_queries = [
        "What are the requirements for machine learning algorithms?",
        "How do deep learning models work?",
        "What is natural language processing?"
    ]
    
    for query in test_queries:
        results = monitored_search.search(query, top_k=5)
        print_query_results(query, results, monitored_search.sparse_retriever.documents)
    
    performance_stats = monitored_search.get_performance_stats()
    print("Performance Statistics:", performance_stats)