"""JSON response shapes. FastAPI uses them for validation + Swagger docs."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class PlateDetection(BaseModel):
    bbox: list[int] = Field(description="[x1, y1, x2, y2] in pixels")
    detection_confidence: float
    plate_text_raw: str = Field(description="Raw CRNN output, kept for audit")
    plate_text: str
    display: str = Field(description="Human-readable, e.g. '123 تونس 4567'")
    format: Literal["TUN", "NT"] | None
    status: Literal["valid", "invalid", "empty"]
    ocr_confidence: float
    uncertain: bool = Field(description="True if ocr_confidence < threshold")


class Timings(BaseModel):
    detection: float
    ocr: float
    total: float


class PredictResponse(BaseModel):
    success: bool
    num_plates: int
    detections: list[PlateDetection]
    timings_ms: Timings


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    models_loaded: bool
    device: str

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=72)   # bcrypt uses at most 72 bytes


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)   # build it directly from a SQLAlchemy User
    id: int
    email: EmailStr
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut