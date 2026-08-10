from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.core.time import now_kst_naive
from app.db.base import Base


class ChatMessage(Base):
    """러닝 최종 리포트 화면의 RAG 챗봇 대화 이력."""

    __tablename__ = "chat_messages"

    message_id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(BigInteger, ForeignKey("running_sessions.session_id", ondelete="CASCADE"), nullable=False)
    user_id    = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(10), nullable=False)  # "user" | "assistant"
    content    = Column(Text, nullable=False)
    sources    = Column(JSONB, nullable=True)  # assistant 메시지가 인용한 논문 청크 메타
    created_at = Column(DateTime, default=now_kst_naive)

    __table_args__ = (
        Index("ix_chat_messages_session_message", "session_id", "message_id"),
    )
