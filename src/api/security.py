"""Password hashing (bcrypt) and JWT tokens."""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_MINUTES = int(os.getenv("TOKEN_MINUTES", "60"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: int) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is missing in .env")
    payload = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> int | None:
    """Return the user id, or None if the token is invalid or expired."""
    try:
        return int(jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


if __name__ == "__main__":
    h = hash_password("MySecret123")
    print("hash:", h[:30] + "...")
    print("good password:", verify_password("MySecret123", h), "| wrong:", verify_password("nope", h))
    t = create_token(42)
    print("token user:", decode_token(t), "| tampered token:", decode_token(t[:-3] + "abc"))