from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:
    @staticmethod
    async def get_by_user_id(db: AsyncSession, user_id: str):
        result = await db.execute(select(User).where(User.user_id == user_id))
        return result.scalars().first()

    @staticmethod
    async def create_user(db: AsyncSession, data: dict) -> User:
        user = User(**data)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
