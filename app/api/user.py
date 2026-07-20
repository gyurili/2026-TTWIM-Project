from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.exceptions import DuplicateUserError, InvalidCredentialsError
from app.core.security import get_current_user
from app.db.session import get_db
from app.repositories import user as user_repo
from app.schemas.common import SuccessResponse
from app.schemas.user import UserCreate, UserOut, UserLogin, TokenResponse, UserInfo
from app.services.user import get_password_hash, verify_password, create_access_token

router = APIRouter()


@router.post("/signup", response_model=SuccessResponse[UserOut])
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    existing = user_repo.get_by_user_id(db, payload.user_id)
    if existing is not None:
        raise DuplicateUserError()

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
    return SuccessResponse(data=user)


@router.post("/login", response_model=SuccessResponse[TokenResponse])
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = user_repo.get_by_user_id(db, payload.user_id)
    if not user or not verify_password(payload.password, user.password_hash):
        raise InvalidCredentialsError()

    token = create_access_token({"sub": str(user.id), "user_id": user.user_id})
    user_info = UserInfo.model_validate(user)
    token_response = TokenResponse(access_token=token, token_type="bearer", user=user_info)
    return SuccessResponse(data=token_response)


@router.get("/me", response_model=SuccessResponse[UserOut])
def get_me(current_user = Depends(get_current_user)):
    return SuccessResponse(data=UserOut.model_validate(current_user))
