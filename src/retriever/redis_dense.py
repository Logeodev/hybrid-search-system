from typing import List, Tuple, Optional

from .base import BaseDenseRetriever
from store import RedisController, to_binary


class RedisDenseRetriever(BaseDenseRetriever):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        index_name: str = "dense_idx",
        index_prefix: str = "doc:",
        vector_dim: Optional[int] = None,
        distance_metric: str = "COSINE",
        create_index: bool = True,
        **model_kwargs,
    ):
        super().__init__(model_name, **model_kwargs)
        self.redis = RedisController(host=redis_host, port=redis_port, db=redis_db)
        self.index_name = index_name
        self.index_prefix = index_prefix
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric
        self.create_index = create_index

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        self.documents = documents
        embeddings = self.model.encode(documents)
        if not embeddings:
            return []

        if self.vector_dim is None:
            self.vector_dim = len(embeddings[0])

        if self.create_index:
            self.redis.create_vector_index(
                self.index_name,
                self.index_prefix,
                self.vector_dim,
                self.distance_metric,
            )

        for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
            key = f"{self.index_prefix}{idx}"
            self.redis.add_document(
                key,
                {
                    "metadata": str(idx),
                    "content": doc,
                    "embedding": to_binary(vector),
                },
            )

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_vector = self._normalize_query_embedding(self.model.encode(query))
        results = self.redis.search_vector(self.index_name, query_vector, top_k)
        return self._convert_results(results)

    def _normalize_query_embedding(self, embedding) -> List[float]:
        if embedding and isinstance(embedding[0], list):
            return embedding[0]
        return embedding

    def _convert_results(self, results: List[Tuple[str, float]]) -> List[Tuple[int, float]]:
        converted: List[Tuple[int, float]] = []
        for key, score in results:
            idx = self._key_to_index(key)
            if idx is None:
                continue
            converted.append((idx, self._normalize_score(score)))
        return converted

    def _key_to_index(self, key: str) -> Optional[int]:
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="ignore")
        if not key.startswith(self.index_prefix):
            return None
        suffix = key[len(self.index_prefix):]
        try:
            return int(suffix)
        except ValueError:
            return None

    def _normalize_score(self, score: float) -> float:
        metric = self.distance_metric.upper()
        if metric == "COSINE":
            return 1.0 - score
        if metric == "L2":
            return -score
        return score
