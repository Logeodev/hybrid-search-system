from typing import Union, List, Literal

from helpers.config import EmbedderConfig
from sentence_transformers import SentenceTransformer
from langchain_openai.embeddings import OpenAIEmbeddings

class Embedder:
    """A class responsible for abstracting any use of external embedding models, such as sentence-transformers or OpenAI's embedding API, ...
    This allows us to easily switch between different embedding providers without changing the rest of the codebase.
    Arguments:
        model_name: The name of the embedding model to use. For example, "all-MiniLM-L6-v2"
        embedding_module: The embedding module to use. Options are 'sentence-transformers', 'local-dmr', or 'openai-api' (see [LangChain's OpenAIEmbeddings](https://docs.langchain.com/oss/python/integrations/text_embedding/openai)).
        kwargs: Additional keyword arguments to pass to the embedding model constructor (e.g., API keys, base URLs, etc.)
    """
    def __init__(self, 
                 model_name: str, 
                 embedding_module: Literal['sentence-transformers', 'local-dmr', 'openai-api'] = 'sentence-transformers', 
                 **kwargs):
        self.model_name = model_name
        self._model = None
        self._embedding_module = embedding_module
        self.__set_model_instance(**kwargs)

    def __set_model_instance(self, **kwargs):
        if self._embedding_module == 'sentence-transformers':
            self._model = SentenceTransformer(self.model_name)
        elif self._embedding_module == 'local-dmr':
            kwargs.setdefault('base_url', "http://localhost:12434/engines/v1")
            kwargs.setdefault('api_key', "some-pass-key")
            self._model = OpenAIEmbeddings(model=self.model_name, **kwargs)
        elif self._embedding_module == 'openai-api':
            self._model = OpenAIEmbeddings(model=self.model_name, **kwargs)

    def encode(self, text: Union[List[str], str]) -> list[float]:
        """Encode a single string or a list of strings into dense vectors."""
        if isinstance(text, str):
            text = [text]
        
        if self._embedding_module == 'sentence-transformers':
            return self._model.encode(text).tolist()
        elif self._embedding_module in ['local-dmr', 'openai-api']:
            return self._model.embed_documents(text)
        # elif hasattr(self._model, 'encode'):
        #     return self._model.encode(text).tolist()
        # elif hasattr(self._model, 'embed_documents'):
        #     return self._model.embed_documents(text)
        # elif hasattr(self._model, 'embed'):
        #     return self._model.embed(text)


if __name__ == "__main__":
    conf = EmbedderConfig()
    print(*conf, end="\n\n")
    embedder = Embedder(**conf)
    print(embedder._model)