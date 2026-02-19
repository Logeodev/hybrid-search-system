# Copilot Instructions for Hybrid Search System

## Project Overview
- Implements a hybrid search system combining dense (embedding-based) and sparse (BM25) retrieval, with score fusion for improved search relevance.
- Main entry point: [src/search/hybrid_rag.py](src/search/hybrid_rag.py) — demonstrates indexing, searching, and fusion.
- Based on [markaicode.com/implement-hybrid-search-rag-performance](https://markaicode.com/implement-hybrid-search-rag-performance).

## Architecture
- **Retriever Layer:**
  - [src/retriever/dense.py](src/retriever/dense.py): Uses SentenceTransformer for dense embeddings.
  - [src/retriever/bm25.py](src/retriever/bm25.py): Implements BM25 with scikit-learn's TfidfVectorizer.
- **Score Fusion:**
  - [src/score/fusion.py](src/score/fusion.py): Combines retrieval scores via Reciprocal Rank Fusion (RRF) or weighted sum.
- **Search Orchestration:**
  - [src/search/hybrid_rag.py](src/search/hybrid_rag.py): Coordinates indexing, searching, and fusion; provides example workflow.

## Key Patterns & Conventions
- All retrievers expose `fit_documents`/`encode_documents` and `search` methods.
- Score fusion methods are static and support both RRF and weighted sum.
- Document indices are used for cross-component communication.
- Example usage and developer workflow are shown in the `demonstrate_hybrid_search()` function.

## Developer Workflows
- **Indexing:** Call `index_documents(documents)` to prepare both dense and sparse indices.
- **Searching:** Use `search(query, top_k)` to retrieve and fuse results.
- **Fusion Method:** Select via `fusion_method` argument (`rrf` or `weighted_sum`).
- **Debugging:** Run [src/search/hybrid_rag.py](src/search/hybrid_rag.py) directly for a full demo.
  - Command: `python -m search.hybrid_rag` (from `src` directory)

## External Dependencies
- See [requirements.txt](requirements.txt) for required packages:
  - `sentence-transformers`, `scikit-learn`, `redis`, `tensorflow<2.11`, `numpy<2`
- Python 3.10 is recommended.

## Integration Points
- Dense and sparse retrievers are independent; fusion logic is centralized in `ScoreFusion`.
- Document text is always referenced by index for consistency.

## Examples
- To add a new retriever, implement `fit_documents` and `search` methods, and update fusion logic if needed.
- To change fusion, modify `fusion_method` in `HybridSearchSystem`.

## References
- [src/retriever/dense.py](src/retriever/dense.py), [src/retriever/bm25.py](src/retriever/bm25.py), [src/score/fusion.py](src/score/fusion.py), [src/search/hybrid_rag.py](src/search/hybrid_rag.py)

