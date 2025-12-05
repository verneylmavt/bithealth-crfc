# app/schemas.py: Request Models (Schemas)
# to define the Pydantic models used in endpoint inputs and keep them separate from logic.

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class DocumentRequest(BaseModel):
    text: str