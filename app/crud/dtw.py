from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.dtw import DtwCalibration
from app.models.user import User
from app.schemas.dtw import DtwData


def get_by_user_id(db: Session, user_id: UUID) -> Optional[DtwCalibration]:
    return db.query(DtwCalibration).filter(DtwCalibration.user_id == user_id).first()


def upsert_calibration(db: Session, user: User, data: DtwData) -> DtwCalibration:
    values = data.model_dump()

    calibration = get_by_user_id(db, user.id)
    if calibration is None:
        calibration = DtwCalibration(user_id=user.id, **values)
        db.add(calibration)
    else:
        for field, value in values.items():
            setattr(calibration, field, value)

    user.is_calibrated = True

    db.commit()
    db.refresh(calibration)
    return calibration
