from datetime import date, datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, constr


class UserCreate(BaseModel):
    user_id: constr(min_length=3, max_length=20)
    nickname: str
    password: constr(min_length=8, max_length=16)
    gender: Literal["man", "woman"]
    birth_date: date  # YYYY-MM-DD 형식으로 파싱
    height: int  # cm
    weight: int  # kg


class UserOut(BaseModel):
    id: UUID
    user_id: str
    nickname: str
    gender: Literal["man", "woman"]
    birth_date: date
    height: int
    weight: int
    is_calibrated: bool = False
    total_running_km: float = 0.0
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    nickname: Optional[str] = None
    gender: Optional[Literal["man", "woman"]] = None
    birth_date: Optional[date] = None
    height: Optional[int] = None
    weight: Optional[int] = None


class UserLogin(BaseModel):
    user_id: str
    password: str


class UserInfo(BaseModel):
    id: UUID
    user_id: str
    nickname: str
    is_calibrated: bool = False

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo
