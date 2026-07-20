from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.running import RunningSessionCreate, RunningSessionOut
from app.crud.running import create_session

router = APIRouter(tags=["Running"])


@router.post("/running/complete", response_model=SuccessResponse[RunningSessionOut])
def complete_running(
    data: RunningSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = create_session(db, current_user.id, data)
    return SuccessResponse(data=session)
