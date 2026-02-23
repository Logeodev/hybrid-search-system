from typing import List, Tuple, Optional

from .base import BaseBM25Retriever
from store import RedisController


class RedisBM25Retriever(BaseBM25Retriever):
    def __init__(
        self,
        *,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        index_name: str = "bm25_idx",
        index_prefix: str = "doc:",
        create_index: bool = True,
        fuzziness: int = 0,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        super().__init__(k1, b)
        self.redis = RedisController(host=redis_host, port=redis_port, db=redis_db)
        self.index_name = index_name
        self.index_prefix = index_prefix
        self.create_index = create_index
        self.fuzziness = fuzziness

    def fit_documents(self, documents: List[str]):
        self.documents = documents

        if self.create_index:
            self.redis.create_text_index(self.index_name, self.index_prefix)

        for idx, doc in enumerate(documents):
            key = f"{self.index_prefix}{idx}"
            self.redis.add_document(
                key,
                {
                    "metadata": str(idx),
                    "content": doc,
                },
            )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        results = self.redis.search_text(
            self.index_name,
            query,
            top_k=top_k,
            fuzziness=self.fuzziness,
            scorer="BM25STD",
        )
        return self._convert_results(results)

    def _convert_results(self, results: List[Tuple[str, float]]) -> List[Tuple[int, float]]:
        converted: List[Tuple[int, float]] = []
        for key, score in results:
            idx = self._key_to_index(key)
            if idx is None:
                continue
            converted.append((idx, score))
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
