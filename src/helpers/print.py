def print_query_results(query: str, results: list[tuple[int, float]], documents: list[str]):
    """Helper function to print search results in a readable format."""
    print(f"Query: {query}")
    print("Top Results:")
    for idx, score in results:
        print(f"  - Doc ID: {idx}, Score: {score:.4f}")
        print(f"    Content: {documents[idx][:200]}...")  # Print first 200 chars of the document