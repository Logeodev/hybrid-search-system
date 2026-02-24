from abc import ABC, abstractmethod
from typing import List, Tuple
from numpy import ndarray
from documents import Document
from .embedder import Embedder

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query:str, top_k:int=5) -> List[Tuple[str, float]]:
        pass

class BaseDenseRetriever(BaseRetriever):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **model_kwargs):
        """Initialize dense retriever with embedding model
        `model_kwargs` are passed to the Embedder class, use **EmbedderConfig() from helpers.config to easily create the config dict.
        """
        self.model = Embedder(model_name, **model_kwargs)
        self.document_embeddings = None
        self.documents: List[Document] = []

    @abstractmethod
    def encode_documents(self, documents: List[Document]):
        pass

class BaseBM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 retriever with tuning parameters.
        k1 controls term frequency saturation, while b controls length normalization.
        Use **BM25Config() from helpers.config to easily create the config dict."""
        self.k1 = k1  # Term frequency saturation point
        self.b = b    # Length normalization factor
        self.documents: List[Document] = []
        self.doc_vectors : ndarray[ndarray] = None
        self.avg_doc_length = 0

    @abstractmethod
    def fit_documents(self, documents: List[Document]):
        pass