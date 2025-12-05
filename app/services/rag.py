# app/services/rag.py: RAG Workflow Service
# to encapsulate the RAG workflow (retrieval + answer) behind RagService, using LangGraph internally.

from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from .embeddings import EmbeddingService
from .document_store import DocumentStore


class RagService:
    """
    High-level RAG service built on LangGraph.

    Responsibilities:
    - Orchestrate retrieval and answer steps.
    - Hide LangGraph details from the API layer.
    - Expose simple methods: answer_question, add_document.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        document_store: DocumentStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._document_store = document_store
        self._chain = self._build_chain()

    def _build_chain(self):
        """
        Build the LangGraph state graph.

        State structure:
        - Input: {"question": <str>}
        - After retrieve: state["context"] = List[str]
        - After answer:   state["answer"] = str
        """
        workflow = StateGraph(dict)

        def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
            question = state["question"]
            query_embedding = self._embedding_service.embed(question)
            context = self._document_store.retrieve(
                query_embedding=query_embedding,
                raw_query=question,
                limit=2,
            )
            state["context"] = context
            return state

        def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
            ctx: List[str] = state.get("context") or []
            if ctx:
                answer = f"I found this: '{ctx[0][:100]}...'"
            else:
                answer = "Sorry, I don't know."
            state["answer"] = answer
            return state

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("answer", answer_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile()

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Public method used by the API layer.

        Returns a dict with the keys:
        - "question"
        - "answer"
        - "context_used"
        """
        state: Dict[str, Any] = {"question": question}
        result_state = self._chain.invoke(state)
        return {
            "question": question,
            "answer": result_state["answer"],
            "context_used": result_state.get("context", []),
        }

    def add_document(self, text: str) -> int:
        """
        Add a document via the embedding + storage pipeline.
        """
        embedding = self._embedding_service.embed(text)
        doc_id = self._document_store.add_document(text, embedding)
        return doc_id

    @property
    def document_store(self) -> DocumentStore:
        return self._document_store

    @property
    def is_ready(self) -> bool:
        return self._chain is not None