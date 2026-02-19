from typing import Union, List, Literal

from sentence_transformers import SentenceTransformer
from langchain_openai.embeddings import OpenAIEmbeddings

class Embedder:
    """A class responsible for abstracting any use of external embedding models, such as sentence-transformers or OpenAI's embedding API, ..."""
    def __init__(self, 
                 model_name: str, 
                 embedding_module: Literal['sentence-transformers', 'local-dmr'] = 'local-dmr', 
                 **kwargs):
        self.model_name = model_name
        self._model = None
        self._embedding_module = embedding_module
        self.__set_model_instance(**kwargs)

    def __set_model_instance(self, **kwargs):
        if self._embedding_module == 'sentence-transformers':
            self._model = SentenceTransformer(self.model_name, **kwargs)
        elif self._embedding_module == 'local-dmr':
            self._model = OpenAIEmbeddings(
                model=self.model_name,
                base_url=kwargs.get('base_url', "http://localhost:12434/engines/v1"),
                api_key=kwargs.get('key', "some-pass-key"),
                **kwargs
                )

    def encode(self, text: Union[List[str], str]) -> list[float]:
        """Encode a single string or a list of strings into dense vectors."""
        if isinstance(text, str):
            text = [text]
        
        if self._embedding_module == 'sentence-transformers':
            return self._model.encode(text).tolist()
        elif self._embedding_module == 'local-dmr':
            return self._model.embed_documents(text)
        # elif hasattr(self._model, 'encode'):
        #     return self._model.encode(text).tolist()
        # elif hasattr(self._model, 'embed_documents'):
        #     return self._model.embed_documents(text)
        # elif hasattr(self._model, 'embed'):
        #     return self._model.embed(text)