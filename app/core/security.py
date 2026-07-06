from typing import Optional

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import UnauthorizedError
from app.db.session import get_db
from app.models.user import User
from app.repositories import user as user_repo

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login", auto_error=False)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if token is None:
        raise UnauthorizedError()

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id = payload.get("user_id")
    except JWTError:
        raise UnauthorizedError()

    if user_id is None:
        raise UnauthorizedError()

    user = user_repo.get_by_user_id(db, user_id)
    if user is None:
        raise UnauthorizedError()

    return user
