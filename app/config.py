# app/config.py: Configuration Module
# to centrally place configuration such as Qdrant URL, collection name, and embedding size.

import os

# URL of the Qdrant instance. If not set, default to local instance.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Name of the Qdrant collection used for this demo.
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "demo_collection")

# Dimensionality of the embedding vectors.
EMBEDDING_DIM = 128