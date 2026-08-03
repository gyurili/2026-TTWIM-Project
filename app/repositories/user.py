from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.user import User


def get_by_user_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.user_id == user_id).first()


def get_by_id(db: Session, id: UUID) -> Optional[User]:
    return db.query(User).filter(User.id == id).first()


def create_user(db: Session, data: dict) -> User:
    user = User(**data)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user(db: Session, user: User, data: dict) -> User:
    for key, value in data.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user: User) -> None:
    db.delete(user)
    db.commit()
