from documents import Document
from typing import Literal

def print_query_results(query: str, results: list[tuple[str, float]], documents: list[Document], mode:Literal['redis', 'integer'] = 'integer'):
    """Helper function to print search results in a readable format."""
    print(f"Query: {query}")
    print("Top Results:")
    for idx, score in results:
        position = int(idx.split(":")[-2]) if mode == 'redis' else int(idx)
        print(f"  - Doc ID: {idx}, Score: {score:.4f}")
        print(f"    Content: {documents[position].text[:200]}...")  # Print first 200 chars of the document