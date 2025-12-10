# Bithealth Associate AI Engineer Take Home Assessment: Code Refactoring

This project contains refactored version of the Python application in `bithealth-crfc/assets/main.py`. The original implementation was a functional but intentionally unstructured single-file RAG (Retrieval-Augmented Generation) demo built using FastAPI, LangGraph, and Qdrant with an in-memory fallback. The objective of this technical test is to redesign existing working code into a clean, maintainable, and testable architecture through proper separation of concerns and object-oriented structuring.

The refactored solution preserves all external behaviours, including the exact /add, /ask, and /status endpoints, while reorganizing the system into clear layers: API handling, RAG workflow logic, embedding generation, and document storage. This modularization eliminates global mutable state, makes dependencies explicit, and prepares the codebase for future extensibility and unit testing.

[Click here to learn more about the project: bithealth-crfc/assets/README.md](https://github.com/verneylmavt/bithealth-crfc/blob/dccd8b2dd3879da12a976d1d83bc1a7281654e66/assets/README.md).

## 📁 Project Structure

```
bithealth-crfc
│
├─ app/
│  ├─ api.py                      # API layer (FastAPI router)
│  ├─ config.py                   # Configuration module
│  ├─ main.py                     # Application entrypoint
│  ├─ schemas.py                  # Request models (schemas)
│  ├─ __init__.py
│  │
│  └─ services/
│     ├─ document_store.py        # Document store abstraction
│     ├─ embeddings.py            # Embedding service
│     ├─ rag.py                   # RAG workflow service
│     └─ __init__.py
│
└─ requirements.txt
```

## ⚖️ Design Decisions

The core goal of the refactor was to reorganize the system around clear software-engineering principles:

- **Encapsulation**: Each concept (embeddings, storage, RAG workflow, API) lives in its own class/module.
- **Separation of Concerns**:
  - The HTTP layer does no computation.
  - The RAG layer knows nothing about FastAPI.
  - The storage layer is completely swappable.
- **Explicit Dependencies**: A single application factory creates all components and wires them together. This eliminates hidden global state and makes control flow predictable.
- **Modularity**: Code is structured so future extensions (new storage backends, real embedding models, additional LangGraph steps) require minimal modification.

## ❓ Trade-Off

A potential simplification was to remove LangGraph entirely and rewrite the retrieval → answer process as a simple, sequential function. The existing workflow is linear and does not strictly require a graph framework.

However, keeping LangGraph has advantages:

- It honours the original system design.
- It preserves compatibility with more complex workflows.
- It demonstrates the ability to encapsulate external frameworks inside clean architectural boundaries.

This trade-off favours extensibility and alignment with the intended evaluation criteria over minimalism.

## 🛠️ Maintainability Improvement

The refactored architecture improves maintainability in several concrete ways:

- The application can now be tested in fine-grained layers without dependence on the web server or Qdrant.
- Modifying or extending one part of the system no longer risks side effects in others.
- The explicit class boundaries reduce cognitive load and make the codebase easier to understand.
- New developers can quickly locate where specific logic lives—embedding logic in embeddings.py, retrieval logic in document_store.py, workflow logic in rag.py, and API routing in api.py.
- The absence of global mutable state prevents hard-to-debug behaviour in concurrent or multi-worker environments.

Overall, this redesign transforms a functional prototype into a maintainable foundation suitable for production-grade systems.

## 🔌 API

1. **Document**  
   `POST /add`: to add a new document to the knowledge base.

   ```bash
    curl -X POST "http://localhost:8000/add" \
    -H "Content-Type: application/json" \
    -d '{"text": "{text}"}'
   ```

2. **Query**  
   `POST /ask`: to run a full retrieval-augmented generation query.

   ```bash
    curl -X POST "http://127.0.0.1:8000/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "{question}"}'
   ```

3. **Status**  
   `GET /status`: to check status of Qdrant, in-memory document, and LangGraph workflow.
   ```bash
    curl "http://127.0.0.1:8000/status"
   ```

## ⚙️ Local Setup

0. Make sure to have the prerequisites:

   - Git
   - Git Large File Storage
   - Python
   - Conda or venv
   - Docker

1. Clone the repository:

   ```bash
    git clone https://github.com/verneylmavt/bithealth-crfc.git
    cd bithealth-crfc
   ```

2. Create environment and install dependencies:

   ```bash
   conda create -n bithealth-crfc python=3.10 -y
   conda activate bithealth-crfc

   pip install -r requirements.txt
   ```

3. Start Docker Desktop and run Qdrant:

   ```bash
   docker run -p 6333:6333 -d qdrant/qdrant
   ```

4. Run the server:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Open the API documentation to make an API call and interact with the app:
   ```bash
   start "http://127.0.0.1:8000/docs"
   ```
