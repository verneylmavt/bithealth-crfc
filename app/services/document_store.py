# app/services/document_store.py: Document Store Abstraction
# to isolate how documents are stored and retrieved.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


@dataclass
class Document:
    """Simple in-memory representation of a stored document."""
    id: int
    text: str
    embedding: List[float]


class DocumentStore(ABC):
    """Abstract interface for storing and retrieving documents."""

    @abstractmethod
    def add_document(self, text: str, embedding: List[float]) -> int:
        """
        Store a document and its embedding.
        Returns an integer document ID.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query_embedding: List[float],
        raw_query: str,
        limit: int = 2,
    ) -> List[str]:
        """
        Retrieve document texts given a query embedding and raw query string.
        Semantics:
        - For Qdrant, use vector search with the given limit.
        - For in-memory, use simple substring matching with fallback.
        """
        raise NotImplementedError

    @property
    def in_memory_count(self) -> int:
        """
        Number of documents stored in memory (0 for persistent backends).
        Used by the /status endpoint to mirror original behaviour.
        """
        return 0

    @property
    def uses_persistent_backend(self) -> bool:
        """
        True if backed by a persistent external storage (like Qdrant),
        False for pure in-memory implementations.
        """
        return False


class InMemoryDocumentStore(DocumentStore):
    """
    Simple in-memory document store.

    Behaviour mirrors the original docs_memory implementation:
    - Substring-based retrieval with fallback to the first document.
    """

    def __init__(self) -> None:
        self._documents: List[Document] = []

    def add_document(self, text: str, embedding: List[float]) -> int:
        # Use index as ID (same pattern as original: len(docs_memory))
        doc_id = len(self._documents)
        doc = Document(id=doc_id, text=text, embedding=embedding)
        self._documents.append(doc)
        return doc_id

    def retrieve(
        self,
        query_embedding: List[float],
        raw_query: str,
        limit: int = 2,
    ) -> List[str]:
        # Embedding is unused here; retrieval is purely substring-based.
        results: List[str] = []
        query_lower = (raw_query or "").lower()

        for doc in self._documents:
            if query_lower in doc.text.lower():
                results.append(doc.text)

        # Fallback: if nothing matched but we have docs, return the first one
        if not results and self._documents:
            results = [self._documents[0].text]

        return results

    @property
    def in_memory_count(self) -> int:
        return len(self._documents)


class QdrantDocumentStore(DocumentStore):
    """
    Document store backed by a Qdrant collection.

    On initialization, the collection is recreated, reproducing the original
    semantics of recreating the collection on startup.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        # Simple monotonically increasing ID generator
        self._next_id = 0

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )

    def add_document(self, text: str, embedding: List[float]) -> int:
        doc_id = self._next_id
        self._next_id += 1

        payload = {"text": text}
        point = PointStruct(id=doc_id, vector=embedding, payload=payload)

        self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
        )

        return doc_id

    # def retrieve(
    #     self,
    #     query_embedding: List[float],
    #     raw_query: str,
    #     limit: int = 2,
    # ) -> List[str]:
    #     hits = self._client.search(
    #         collection_name=self._collection_name,
    #         query_vector=query_embedding,
    #         limit=limit,
    #     )
    #     # hits = self._client.query_points(
    #     #     collection_name=self._collection_name,
    #     #     query_vector=query_embedding,
    #     #     limit=limit,
    #     # )
    #     return [hit.payload["text"] for hit in hits]
    
    # def retrieve(self, query_embedding: Sequence[float], limit: int = 2) -> List[str]:
    #     """
    #     Retrieve the top-k most similar documents given a query embedding.

    #     Supports both:
    #     - Newer Qdrant client: `query_points`
    #     - Older Qdrant client: `search`
    #     """
    #     # Newer Qdrant (>= 1.12): unified query API
    #     if hasattr(self._client, "query_points"):
    #         response = self._client.query_points(
    #             collection_name=self._collection_name,
    #             query=list(query_embedding),   # dense vector
    #             limit=limit,
    #             with_payload=True,
    #         )
    #         hits = getattr(response, "points", [])
    #     # Older Qdrant: still had `search`
    #     elif hasattr(self._client, "search"):
    #         hits = self._client.search(
    #             collection_name=self._collection_name,
    #             query_vector=list(query_embedding),
    #             limit=limit,
    #             with_payload=True,
    #         )
    #     else:
    #         # Extremely unlikely, but explicit failure is better than a cryptic AttributeError
    #         raise RuntimeError(
    #             "QdrantClient has neither `query_points` nor `search`. "
    #             "Please check the installed qdrant-client version."
    #         )

    #     texts: List[str] = []
    #     for hit in hits:
    #         payload = getattr(hit, "payload", None)
    #         if isinstance(payload, dict) and "text" in payload:
    #             texts.append(str(payload["text"]))
    #     return texts
    
    def retrieve(
        self,
        query_embedding: List[float],
        raw_query: str,          # <--- add this parameter back
        limit: int = 2,
    ) -> List[str]:
        """
        Retrieve the top-k most similar documents given a query embedding.

        Supports both:
        - Newer Qdrant client: `query_points`
        - Older Qdrant client: `search`

        Note: raw_query is accepted for interface compatibility, but unused here.
        """
        # Newer Qdrant (>= 1.10): unified query API
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=list(query_embedding),   # dense vector
                limit=limit,
                with_payload=True,
            )
            hits = getattr(response, "points", [])
        # Older Qdrant: still has `search`
        elif hasattr(self._client, "search"):
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=list(query_embedding),
                limit=limit,
                with_payload=True,
            )
        else:
            # Extremely unlikely, but explicit failure is better than a cryptic AttributeError
            raise RuntimeError(
                "QdrantClient has neither `query_points` nor `search`. "
                "Please check the installed qdrant-client version."
            )

        texts: List[str] = []
        for hit in hits:
            payload = getattr(hit, "payload", None)
            if isinstance(payload, dict) and "text" in payload:
                texts.append(str(payload["text"]))
        return texts

    @property
    def uses_persistent_backend(self) -> bool:
        return True


def create_document_store(
    qdrant_url: str,
    collection_name: str,
    vector_size: int,
) -> DocumentStore:
    """
    Try to create a Qdrant-backed store. If anything fails, fall back to in-memory.

    This preserves the original behaviour of:
    - "Trying Qdrant"
    - Printing a warning and falling back to a list if unavailable.
    """
    try:
        client = QdrantClient(qdrant_url)
        store: DocumentStore = QdrantDocumentStore(
            client=client,
            collection_name=collection_name,
            vector_size=vector_size,
        )
        return store
    except Exception:
        print("Qdrant not available. Falling back to in-memory list.")
        return InMemoryDocumentStore()