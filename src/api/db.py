"""PostgreSQL connection with SQLAlchemy (users, feedbacks...)."""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tlpr:tlpr_dev@localhost:5433/tlpr")
# SQLAlchemy needs to know which driver to use: psycopg (v3)
engine = create_engine(DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1),
                       pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: one DB session per request, always closed at the end."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from src.api import models  # noqa: F401  (registers the tables)
    Base.metadata.create_all(engine)