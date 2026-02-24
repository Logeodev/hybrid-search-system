from dataclasses import dataclass
from typing import List

@dataclass
class Document:
    """Base class for documents"""
    idx: int
    text: str
    embedding: List[float] = None
    chunk:int = 0