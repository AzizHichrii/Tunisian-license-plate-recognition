"""Valley: RAG chatbot about cars and Tunisian road law (logged-in users only)."""
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_current_user
from src.api.models import User
from src.api.schemas import ValleyRequest, ValleyResponse
from src.rag.chain import ask

router = APIRouter(prefix="/valley", tags=["valley"])


@router.post("/ask", response_model=ValleyResponse)
def valley_ask(body: ValleyRequest, user: User = Depends(get_current_user)):
    try:
        result = ask(body.question.strip())
    except RuntimeError as e:   # every Gemini model unavailable / quota reached
        raise HTTPException(503, "Valley is busy right now, please try again in a minute.") from e
    return ValleyResponse(**{k: result[k] for k in ("answer", "language", "model", "sources", "timings_ms")})
