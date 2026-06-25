import json
from sqlalchemy.orm import Session
from typing import List
from app.models.imu import GaitRecord, GaitSession
from app.schemas.imu import GaitRecordCreate


def bulk_create_gait_records(
    db: Session,
    records: List[GaitRecordCreate],
    session_id: str,
) -> int:
    if not records:
        return 0

    first = records[0]

    db_session = GaitSession(
        id    = session_id,
        label = first.label,
    )
    db.add(db_session)
    db.flush()

    db_records = [
        GaitRecord(
            session_id   = session_id,
            window_index = r.window_index,
            start_time   = r.start_time,
            end_time     = r.end_time,
            label        = r.label,
            subject      = r.subject,
            raw_acc_x    = json.dumps(r.raw_acc_x),
            raw_acc_y    = json.dumps(r.raw_acc_y),
            raw_acc_z    = json.dumps(r.raw_acc_z),
            raw_gyro_x   = json.dumps(r.raw_gyro_x),
            raw_gyro_y   = json.dumps(r.raw_gyro_y),
            raw_gyro_z   = json.dumps(r.raw_gyro_z),
        )
        for r in records
    ]

    db.bulk_save_objects(db_records)
    db.commit()
    return len(db_records)
