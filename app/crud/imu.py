from sqlalchemy.orm import Session
from typing import List
from app.models.imu import IMURecord
from app.schemas.imu import IMURecordCreate


def create_imu_record(db: Session, record: IMURecordCreate) -> IMURecord:
    db_record = IMURecord(**record.model_dump())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


def bulk_create_imu_records(db: Session, records: List[IMURecordCreate]) -> int:
    """배치 삽입 — SQLite 덤프 전체를 한 번에 저장"""
    db_records = [IMURecord(**r.model_dump()) for r in records]
    db.bulk_save_objects(db_records)
    db.commit()
    return len(db_records)