from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.base import CamelModel

_SNIPPET_MAX_CHARS = 120


class ChatAskIn(CamelModel):
    question: str = Field(min_length=1, max_length=500)


class ChatSourceOut(CamelModel):
    doc_id: str
    title: str
    page: int
    snippet: str


class ChatMessageOut(CamelModel):
    message_id: int
    role: Literal["user", "assistant"]
    content: str
    sources: list[ChatSourceOut] | None = None
    created_at: datetime | None = None


class ChatAnswerOut(CamelModel):
    message_id: int
    answer: str
    sources: list[ChatSourceOut] = []


class ChatHistoryOut(CamelModel):
    messages: list[ChatMessageOut]
