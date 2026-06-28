from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
    @staticmethod
    def get_by_user_id(db: Session, user_id: str):
        return db.query(User).filter(User.user_id == user_id).first()

    @staticmethod
    def create_user(db: Session, data: dict) -> User:
        user = User(**data)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
