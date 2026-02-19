from typing import Dict, Any

class PerformanceMonitor:
    """Monitor hybrid search performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'queries_processed': 0,
            'avg_response_time': 0,
            'cache_hits': 0,
            'total_time': 0
        }
    
    def record_query(self, query_time: float, cache_hit: bool = False):
        """Record performance metrics for a query"""
        self.metrics['queries_processed'] += 1
        self.metrics['total_time'] += query_time
        
        if cache_hit:
            self.metrics['cache_hits'] += 1
        
        # Update average response time
        self.metrics['avg_response_time'] = (
            self.metrics['total_time'] / self.metrics['queries_processed']
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance summary"""
        cache_rate = (
            self.metrics['cache_hits'] / self.metrics['queries_processed'] 
            if self.metrics['queries_processed'] > 0 else 0
        )
        
        return {
            'total_queries': self.metrics['queries_processed'],
            'average_response_time_ms': self.metrics['avg_response_time'] * 1000,
            'cache_hit_rate': cache_rate,
            'queries_per_second': 1 / self.metrics['avg_response_time'] if self.metrics['avg_response_time'] > 0 else 0
        }
