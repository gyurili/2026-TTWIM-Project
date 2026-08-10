from uuid import UUID

from sqlalchemy.orm import Session

from app.models.chat import ChatMessage


def get_messages_by_session(db: Session, session_id: int, limit: int | None = None) -> list[ChatMessage]:
    query = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.message_id.asc())
    )
    if limit is not None:
        # 최근 N개가 필요한 경우: 뒤에서부터 잘라야 하므로 desc로 가져온 뒤 뒤집는다.
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.message_id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))
    return query.all()


def create_messages(
    db: Session,
    session_id: int,
    user_id: UUID,
    pairs: list[tuple[str, str, list[dict] | None]],
) -> list[ChatMessage]:
    """pairs: [(role, content, sources), ...]. user+assistant를 한 트랜잭션에 함께 커밋해
    LLM 실패로 질문만 저장되는 상황을 막는다."""
    created = []
    for role, content, sources in pairs:
        msg = ChatMessage(session_id=session_id, user_id=user_id, role=role, content=content, sources=sources)
        db.add(msg)
        created.append(msg)
    db.commit()
    for msg in created:
        db.refresh(msg)
    return created


def delete_messages_by_session(db: Session, session_id: int) -> int:
    count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.commit()
    return count
