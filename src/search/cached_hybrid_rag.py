import redis, pickle, hashlib
from typing import List, Tuple

from .hybrid_rag import HybridSearchSystem

class CachedHybridSearch(HybridSearchSystem):
    """Hybrid search with Redis caching of search results"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, cache_hour_duration:float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.cache_ttl = int(cache_hour_duration * 60 * 60)  # 60*60 = 3600 = 1 hour cache TTL
    
    def _generate_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        return f"hybrid_search:{query_hash}"
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search with caching"""
        cache_key = self._generate_cache_key(query, top_k)
        
        # Try to get from cache
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
        except Exception as e:
            print(f"Cache read error: {e}")
        
        # Perform search
        results = super().search(query, top_k)
        
        # Cache results
        try:
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                pickle.dumps(results)
            )
        except Exception as e:
            print(f"Cache write error: {e}")
        
        return results
    
    def invalidate_cache(self, pattern: str = "hybrid_search:*"):
        """Clear search cache"""
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)


if __name__ == "__main__":
    # Sample usage
    from ._samples import documents
    from helpers.config import EmbedderConfig, HybridSearchConfig
    from os import getenv
    
    try:
        redis.Redis(retry=False).ping()
    except redis.exceptions.ConnectionError as e:
        raise Exception(f"Could not connect to Redis server: you need to create a Redis server and ensure it's running. Error details: {e}")
    except Exception as e:
        print(f"Redis connection error: {e}")

    embed_conf = EmbedderConfig(
        model_name=getenv("EMBEDDING_MODEL"), embedding_module='local-dmr')
    search_conf = HybridSearchConfig(dense_weight=0.6, sparse_weight=0.4)
    cached_search = CachedHybridSearch(config=search_conf, embedder_config=embed_conf)
    cached_search.index_documents(documents)
    
    query = "What is the capital of France?"
    results = cached_search.search(query, top_k=5)
    print(f"Search results for '{query}': {results}")