"""Feedbacks: logged-in users post (text + 1-5 stars); everyone can read."""
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from src.api.db import get_db
from src.api.deps import get_current_user
from src.api.models import Feedback, User
from src.api.schemas import FeedbackCreate, FeedbackList, FeedbackOut

router = APIRouter(prefix="/feedback", tags=["feedback"])


def to_out(f: Feedback) -> FeedbackOut:
    return FeedbackOut(id=f.id, rating=f.rating, text=f.text, username=f.user.username,
                       created_at=f.created_at)


@router.post("", response_model=FeedbackOut, status_code=201)
def create_feedback(body: FeedbackCreate, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    fb = Feedback(user_id=user.id, rating=body.rating, text=body.text.strip())
    db.add(fb)
    db.commit()
    fb.user = user
    return to_out(fb)


@router.get("", response_model=FeedbackList)
def list_feedback(limit: int = 50, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    items = db.scalars(select(Feedback).options(joinedload(Feedback.user))
                       .order_by(Feedback.created_at.desc()).limit(limit)).all()
    count, avg = db.execute(select(func.count(Feedback.id), func.avg(Feedback.rating))).one()
    return FeedbackList(count=count, average=round(float(avg), 2) if avg is not None else None,
                        items=[to_out(f) for f in items])
