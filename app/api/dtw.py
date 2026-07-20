from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.exceptions import DtwNotFoundError
from app.core.security import get_current_user
from app.crud import dtw as dtw_crud
from app.db.session import get_db
from app.models.user import User
from app.schemas.common import SuccessResponse
from app.schemas.dtw import DtwData

router = APIRouter(prefix="/dtw", tags=["DTW"])


@router.post("/", response_model=SuccessResponse[DtwData])
def save_dtw(
    data: DtwData,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    calibration = dtw_crud.upsert_calibration(db, current_user, data)
    return SuccessResponse(data=DtwData.model_validate(calibration))


@router.get("/me", response_model=SuccessResponse[DtwData])
def get_my_dtw(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    calibration = dtw_crud.get_by_user_id(db, current_user.id)
    if calibration is None:
        raise DtwNotFoundError()
    return SuccessResponse(data=DtwData.model_validate(calibration))
