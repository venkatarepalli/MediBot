from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

from src.rbac.access_config import get_access_roles, can_use_sql_rag, COLLECTIONS, validate_role
from src.api.auth import authenticate, create_token, get_current_role
from src.api.routing import classify_question_type
from src.ingestion.embed_and_index import get_qdrant_client
from src.retrieval.hybrid_rerank import hybrid_search, rerank
from src.retrieval.generate_answer import generate_answer
from src.sql_rag.sql_rag_chain import sql_rag_chain

app = FastAPI(title="MediBot API")
qdrant_client = get_qdrant_client()


class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    role: str

class ChatRequest(BaseModel):
    question: str

class Source(BaseModel):
    source_document: str
    section_title: str
    collection: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_type: str
    role: str

@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    role = authenticate(req.username, req.password)
    return LoginResponse(token=create_token(req.username, role), role=role)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, role: str = Depends(get_current_role)):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    question_type = classify_question_type(req.question)

    if question_type == "analytical":
        if not can_use_sql_rag(role):
            return ChatResponse(
                answer=f"As a {role}, I don't have access to analytical/operational data queries.",
                sources=[], retrieval_type="sql_rag", role=role,
            )
        try:
            answer = sql_rag_chain(req.question)
        except Exception:
            answer = "I wasn't able to process that as a data query — try rephrasing, or ask a document-based question instead."
        return ChatResponse(answer=answer, sources=[], retrieval_type="sql_rag", role=role)

    candidates = hybrid_search(qdrant_client, role, req.question, prefetch_limit=10)
    top_chunks = rerank(req.question, candidates, top_n=3)
    allowed_collections = [c for c in COLLECTIONS if role in get_access_roles(c)]
    answer, refused = generate_answer(req.question, role, allowed_collections, top_chunks)

    sources = [] if refused else [
        Source(source_document=c["source_document"], section_title=c["section_title"], collection=c["collection"])
        for c in top_chunks
    ]
    return ChatResponse(answer=answer, sources=sources, retrieval_type="hybrid_rag", role=role)

@app.get("/collections/{role}")
def get_collections(role: str):
    try:
        validate_role(role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")
    return {"role": role, "collections": [c for c in COLLECTIONS if role in get_access_roles(c)]}


@app.get("/health")
def health():
    return {"status": "ok"}