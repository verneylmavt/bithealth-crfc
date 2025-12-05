# app/main.py: Application Entrypoint
# to wire everything together and expose the FastAPI app object.

from fastapi import FastAPI

from .config import QDRANT_URL, QDRANT_COLLECTION_NAME, EMBEDDING_DIM
from .services.embeddings import FakeEmbeddingService
from .services.document_store import create_document_store
from .services.rag import RagService
from .api import create_router


def create_app() -> FastAPI:
    """
    Application factory.

    Responsibilities:
    - Instantiate infrastructure (embedding service, document store).
    - Instantiate RagService with explicit dependencies.
    - Create the FastAPI app and attach the router.
    """
    embedding_service = FakeEmbeddingService(dim=EMBEDDING_DIM)
    document_store = create_document_store(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION_NAME,
        vector_size=EMBEDDING_DIM,
    )
    rag_service = RagService(embedding_service, document_store)

    app = FastAPI(title="Learning RAG Demo")
    router = create_router(rag_service)
    app.include_router(router)

    return app


# This is the object uvicorn will look for: `uvicorn app.main:app`
app = create_app()
