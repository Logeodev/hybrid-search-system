# Hybrid Search System

Implementation for a hybrid search system, based on text search and RAG hybrid retrieval of search results. 

Using https://markaicode.com/implement-hybrid-search-rag-performance/ as a base.


## TODO 

Redis :
[tips](https://medium.com/@Nexumo_/8-redis-vector-index-tips-for-low-latency-retrieval-2dec2ab4008a)
- Implement a generic document parser, with support for PDF, HTML, and plain text, as a first step
- Interface doc parsing with ingestion (langchain's `Document` ?)

Doc ingestion :
- Clever use of langchain's text splitters, to split documents into chunks of a certain size, with some overlap, and store them in the database with their metadata (source, page number, etc.)
- See for [NLM Ingestor implementation](https://github.com/nlmatics/nlm-ingestor), handles also OCR and PDF parsing, as a docker service ?