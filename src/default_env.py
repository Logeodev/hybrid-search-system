import os

# When working with local development
# Use of Docker Model Runner (DMR) for embedding generation
os.environ.setdefault("EMBEDDING_MODEL", "docker.io/embeddinggemma:300M-Q8_0")