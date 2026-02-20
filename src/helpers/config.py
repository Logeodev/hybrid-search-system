from typing import Literal
from dataclasses import dataclass, fields
from collections.abc import Mapping

class ConfigObject(Mapping):
    def __getitem__(self, key):
        return getattr(self, key)
    def __iter__(self):
        return (f.name for f in fields(self))
    def __len__(self):
        return len(fields(self))
    
@dataclass
class EmbedderConfig(ConfigObject):
    model_name: str = "all-MiniLM-L6-v2"
    embedding_module: Literal['sentence-transformers', 'local-dmr', 'openai-api'] = 'sentence-transformers'

@dataclass
class BM25Config(ConfigObject):
    k1: float = 1.2
    b: float = 0.75

@dataclass
class HybridSearchConfig(ConfigObject):
    fusion_method: Literal["rrf", "weighted_sum"] = "rrf"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3