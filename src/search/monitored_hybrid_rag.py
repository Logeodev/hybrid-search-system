from typing import List, Tuple, Dict, Any
import time
from search import HybridSearchSystem
from score import PerformanceMonitor

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