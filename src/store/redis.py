import redis, struct
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType

def to_binary(vector):
    return struct.pack('>' + 'f'*len(vector), *vector)

class RedisController:
    def __init__(self, host:str="localhost", port:int=6379, db:int=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def set(self, key, value, *kwargs):
        self.redis_client.set(key, value, *kwargs)

    def get(self, key):
        return self.redis_client.get(key)

    def delete(self, key):
        self.redis_client.delete(key)

    def exists(self, key):
        return self.redis_client.exists(key)
    
    def add_document(self, key:str, mapping:dict[str, str]):
        self.redis_client.hset(key, mapping=mapping)

    def search_vector(self, index_name:str, query_vector:list[float], top_k=10):
        # Search for similar vectors using RediSearch
        query = f'*=>[KNN {top_k} @embedding $vec AS score]'
        params = {"vec": to_binary(query_vector)}
        
        try:
            results = self.redis_client.ft(index_name).search(query, query_params=params)
            return [(res.id, res.score) for res in results.docs]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
        
    def search_text(
        self,
        index_name: str,
        query_text: str,
        top_k: int = 10,
        fuzziness: int = 0,
        scorer: str = "BM25STD",
    ):
        # Search for similar text using RediSearch's fuzzy matching
        if fuzziness < 0 or fuzziness > 3:
            raise ValueError("Fuzziness must be between 0 and 3")
        
        fuzzy_wildcard = "" #if fuzziness == 0 else "%"*fuzziness
        query = f'@content:({fuzzy_wildcard}{query_text}{fuzzy_wildcard})'
        
        search_query = Query(query).paging(0, top_k)
        if scorer:
            search_query = search_query.scorer(scorer).with_scores()

        try:
            results = self.redis_client.ft(index_name).search(search_query)
            return [(res.id, res.score) for res in results.docs]
        except Exception as e:
            print(f"Text search error: {e}")
            return []

    def create_vector_index(self, index_name:str, index_prefix:str, vector_dim:int, distance_metric="COSINE"):
        # Create RediSearch index for vector search
        fields = (
            TextField("metadata"),
            TextField("content"), 
            VectorField(
                "embedding", 
                "HNSW", {
                    "TYPE": "FLOAT32", 
                    "DIM": vector_dim, 
                    "DISTANCE_METRIC": distance_metric
                    }
                )
            )
        definition = IndexDefinition(
            prefix=[index_prefix], 
            index_type=IndexType.HASH
        )

        try:
            self.redis_client.ft(index_name).create_index(
                fields=fields,
                definition=definition
            )
        except Exception as e:
            print(f"Index creation error: {e}")

    def create_text_index(self, index_name: str, index_prefix: str):
        # Create RediSearch index for text search
        fields = (
            TextField("metadata"),
            TextField("content"),
        )
        definition = IndexDefinition(
            prefix=[index_prefix],
            index_type=IndexType.HASH,
        )

        try:
            self.redis_client.ft(index_name).create_index(
                fields=fields,
                definition=definition,
            )
        except Exception as e:
            print(f"Index creation error: {e}")

    # def remove_documents()