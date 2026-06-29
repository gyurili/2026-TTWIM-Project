from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories import user as user_repo
from app.schemas.user import UserCreate, UserOut, UserLogin, TokenResponse, UserInfo
from app.services.user import get_password_hash, verify_password, create_access_token

router = APIRouter()


@router.post("/signup", response_model=UserOut)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    existing = user_repo.get_by_user_id(db, payload.user_id)
    if existing is not None:
        raise HTTPException(status_code=400, detail="User ID already exists")

    user_data = {
        "user_id": payload.user_id,
        "nickname": payload.nickname,
        "password_hash": get_password_hash(payload.password),
        "gender": payload.gender,
        "birth_date": payload.birth_date,
        "height": payload.height,
        "weight": payload.weight,
    }

    user = user_repo.create_user(db, user_data)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = user_repo.get_by_user_id(db, payload.user_id)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect user ID or password")

    token = create_access_token({"sub": str(user.id), "user_id": user.user_id})
    user_info = UserInfo(id=str(user.id), user_id=user.user_id, nickname=user.nickname)
    return {"access_token": token, "token_type": "bearer", "user": user_info}
