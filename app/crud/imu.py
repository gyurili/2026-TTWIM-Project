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
        id      = session_id,
        user_id = first.user_id,
        label   = first.label,
        weight  = first.weight,
        height  = first.height,
    )
    db.add(db_session)
    db.flush()

    db_records = [
        GaitRecord(
            session_id       = session_id,
            side             = r.side,
            step_index       = r.step_index,
            start_time       = r.start_time,
            end_time         = r.end_time,
            peak_acc_z       = r.peak_acc_z,
            cadence          = r.cadence,
            contact_time     = r.contact_time,
            gait_label       = r.label,
            resampled_acc_x  = json.dumps(r.resampled_acc_x),
            resampled_acc_y  = json.dumps(r.resampled_acc_y),
            resampled_acc_z  = json.dumps(r.resampled_acc_z),
            resampled_gyro_x = json.dumps(r.resampled_gyro_x),
            resampled_gyro_y = json.dumps(r.resampled_gyro_y),
            resampled_gyro_z = json.dumps(r.resampled_gyro_z),
        )
        for r in records
    ]

    db.bulk_save_objects(db_records)
    db.commit()
    return len(db_records)