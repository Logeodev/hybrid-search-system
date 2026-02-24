from typing import List, Tuple, Optional

from .base import BaseDenseRetriever
from documents import Document
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

    def encode_documents(self, documents: List[Document]) -> List[List[float]]:
        self.documents = documents
        texts = [doc.text for doc in documents]
        embeddings = self.model.encode(texts)
        
        if not embeddings:
            return []
        
        for e, d in zip(embeddings, documents):
            d.embedding = e

        if self.vector_dim is None:
            self.vector_dim = len(embeddings[0])

        if self.create_index:
            self.redis.create_vector_index(
                self.index_name,
                self.index_prefix,
                self.vector_dim,
                self.distance_metric,
            )

        for idx, doc in enumerate(documents):
            key = f"{self.index_prefix}:{doc.idx}:{doc.chunk}"
            self.redis.add_document(
                key,
                {
                    "metadata": f"{idx}/{doc.idx}/{doc.chunk}",
                    "content": doc.text,
                    "embedding": to_binary(doc.embedding),
                },
            )

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_vector = self._normalize_query_embedding(self.model.encode(query))
        results = self.redis.search_vector(self.index_name, query_vector, top_k)
        return results

    def _normalize_query_embedding(self, embedding) -> List[float]:
        if embedding and isinstance(embedding[0], list):
            return embedding[0]
        return embedding

    def _normalize_score(self, score: float) -> float:
        metric = self.distance_metric.upper()
        if metric == "COSINE":
            return 1.0 - score
        if metric == "L2":
            return -score
        return score
