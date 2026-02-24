import default_env

if __name__ == "__main__":
    from os import getenv
    from search.hybrid_rag import HybridSearchSystem
    from retriever import RedisBM25Retriever, RedisDenseRetriever
    from documents import preprocess_documents
    from helpers.print import print_query_results
    from helpers.config import HybridSearchConfig

    bm25_retriever = RedisBM25Retriever(fuzziness=3)
    dense_retriever = RedisDenseRetriever(model_name=getenv('EMBEDDING_MODEL'), embedding_module='local-dmr')

    search = HybridSearchSystem(
        dense_retriever=dense_retriever,
        sparse_retriever=bm25_retriever,
        config=HybridSearchConfig(dense_weight=0.7, sparse_weight=0.3, fusion_method="weighted_sum")
    )

    # Ingestion 
    # Bring some dataset in project's root : `data.csv`
    # Install dependencies as needed: pip install pandas
    # I used https://www.kaggle.com/datasets/younushassankhan/python-faqs dataset
    import pandas as pd

    file_path = "../data.csv"
    df = pd.read_csv(file_path, encoding='iso-8859-2')

    print("Sample 5 records:")
    print(df.sample(5))
    print(df.shape)

    ## Preprocess documents
    docs = preprocess_documents((df['Question'] + df['Answer']).tolist())

    ## Index documents in Redis
    search.index_documents(docs)

    query = "What is Python used for ?"
    res = search.search(query, top_k=5)
    print("Search results:")
    print_query_results(query, res, search.documents, mode='redis')