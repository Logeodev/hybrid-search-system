from abc import ABC
from typing import List, Tuple

class BaseRetriver(ABC):

    def search(self, query:str, top_k:int=5) -> List[Tuple[int, float]]:
        pass