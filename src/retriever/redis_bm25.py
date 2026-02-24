from typing import List, Tuple, Optional

from .base import BaseBM25Retriever
from documents import Document
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

    def fit_documents(self, documents: List[Document]):
        self.documents = documents

        if self.create_index:
            self.redis.create_text_index(self.index_name, self.index_prefix)

        for idx, doc in enumerate(documents):
            key = f"{self.index_prefix}:{doc.idx}:{doc.chunk}"
            self.redis.add_document(
                key,
                {
                    "metadata": f"{idx}/{doc.idx}/{doc.chunk}",
                    "content": doc.text,
                },
            )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        results = self.redis.search_text(
            self.index_name,
            query,
            top_k=top_k,
            fuzziness=self.fuzziness,
            scorer="BM25STD",
        )
        return results
