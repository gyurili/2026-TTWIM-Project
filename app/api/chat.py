from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.exceptions import SessionForbiddenError, SessionNotFoundError
from app.core.security import get_current_user
from app.crud.chat import delete_messages_by_session, get_messages_by_session
from app.crud.running import get_session
from app.db.session import get_db
from app.models.running import RunningSession
from app.models.user import User
from app.schemas.chat import ChatAnswerOut, ChatAskIn, ChatHistoryOut, ChatMessageOut
from app.schemas.common import SuccessResponse
from app.services.chat import answer_question

router = APIRouter(tags=["Chat"])


def _get_owned_session(db: Session, session_id: int, user: User) -> RunningSession:
    session = get_session(db, session_id)
    if not session:
        raise SessionNotFoundError()
    if session.user_id != user.id:
        raise SessionForbiddenError()
    return session


@router.post("/running/{session_id}/chat", response_model=SuccessResponse[ChatAnswerOut])
def ask_chat(
    session_id: int,
    data: ChatAskIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _get_owned_session(db, session_id, current_user)
    result = answer_question(db, session, current_user, data.question)
    return SuccessResponse(data=result)


@router.get("/running/{session_id}/chat", response_model=SuccessResponse[ChatHistoryOut])
def get_chat_history(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _get_owned_session(db, session_id, current_user)
    messages = get_messages_by_session(db, session.session_id)
    return SuccessResponse(data=ChatHistoryOut(messages=[ChatMessageOut.model_validate(m) for m in messages]))


@router.delete("/running/{session_id}/chat", response_model=SuccessResponse[dict])
def reset_chat(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _get_owned_session(db, session_id, current_user)
    deleted = delete_messages_by_session(db, session.session_id)
    return SuccessResponse(data={"deletedCount": deleted})
