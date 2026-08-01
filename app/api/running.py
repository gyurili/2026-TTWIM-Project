from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exceptions import SessionForbiddenError, SessionNotFoundError
from app.core.security import get_current_user
from app.crud.running import create_session, get_session
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.running import RunningFeedbackOut, RunningSessionCreate, RunningSessionOut
from app.services.feedback import generate_session_feedback

router = APIRouter(tags=["Running"])


@router.post("/running/complete", response_model=SuccessResponse[RunningSessionOut])
def complete_running(
    data: RunningSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = create_session(db, current_user.id, data)
    return SuccessResponse(data=session)


@router.post("/running/{session_id}/feedback", response_model=SuccessResponse[RunningFeedbackOut])
def create_session_feedback(
    session_id: int,
    force_regenerate: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = get_session(db, session_id)
    if not session:
        raise SessionNotFoundError()
    if session.user_id != current_user.id:
        raise SessionForbiddenError()
    result = generate_session_feedback(db, session, current_user.id, current_user, force_regenerate)
    return SuccessResponse(data=result)
