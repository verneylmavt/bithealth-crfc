# app/api.py: FastAPI Router
# to define the HTTP endpoints and translate between HTTP and the RagService. 
# The router is constructed with a RagService instance, so it is easy to test or replace.

import time

from fastapi import APIRouter, HTTPException

from .schemas import QuestionRequest, DocumentRequest
from .services.rag import RagService


def create_router(rag_service: RagService) -> APIRouter:
    """
    Create an APIRouter wired to a specific RagService instance.

    This avoids global variables and keeps web concerns separate from
    business logic.
    """
    router = APIRouter()

    @router.post("/ask")
    def ask_question(req: QuestionRequest):
        start = time.time()
        try:
            result = rag_service.answer_question(req.question)
            latency = round(time.time() - start, 3)
            return {
                "question": result["question"],
                "answer": result["answer"],
                "context_used": result["context_used"],
                "latency_sec": latency,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/add")
    def add_document(req: DocumentRequest):
        try:
            doc_id = rag_service.add_document(req.text)
            return {"id": doc_id, "status": "added"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/status")
    def status():
        store = rag_service.document_store
        return {
            "qdrant_ready": store.uses_persistent_backend,
            "in_memory_docs_count": store.in_memory_count,
            "graph_ready": rag_service.is_ready,
        }

    return router