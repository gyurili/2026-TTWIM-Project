from sqlalchemy.orm import Session
from typing import List
from app.models.imu import IMURecord, IMUSession
from app.schemas.imu import IMURecordCreate

# 단건 생성은 안 쓰이거나 테스트용이겠지만, 구조에 맞춰 수정해 둡니다.
def create_imu_record(db: Session, record: IMURecordCreate) -> IMURecord:
    # 1. 세션이 없을 때를 대비한 방어 코드 (단건 생성 시 필요)
    if record.session_id:
        session_exists = db.query(IMUSession).filter(IMUSession.id == record.session_id).first()
        if not session_exists:
            db_session = IMUSession(
                id=record.session_id,
                user_id=record.user_id,
                label=record.label,
                weight=record.weight,
                height=record.height
            )
            db.add(db_session)
            db.flush()

    # 2. 레코드 저장
    db_record = IMURecord(
        session_id=record.session_id,
        side=record.side,
        acc_x=record.acc_x, acc_y=record.acc_y, acc_z=record.acc_z,
        gyro_x=record.gyro_x, gyro_y=record.gyro_y, gyro_z=record.gyro_z,
        created_at=record.created_at
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record





def bulk_create_imu_records(db: Session, records: List[IMURecordCreate], session_id: str) -> int:
    """배치 삽입 — 첫 번째 레코드에서 세션 정보를 추출해 저장 후 벌크 인서트"""
    if not records:
        return 0
    
    # 1. 세션 생성 (이건 첫 데이터 기준으로 1번만 만드는 게 맞습니다)
    first = records[0]
    db_session = IMUSession(
        id=session_id,
        user_id=first.user_id,
        label=first.label,
        weight=first.weight,
        height=first.height
    )
    db.add(db_session)
    db.flush()

    # 2. 순수 IMU 센서 데이터 배열 생성 (여기서 각 레코드 고유의 시간을 넣어줍니다)
    db_records = [
        IMURecord(
            session_id=session_id,
            side=r.side,
            acc_x=r.acc_x, acc_y=r.acc_y, acc_z=r.acc_z,
            gyro_x=r.gyro_x, gyro_y=r.gyro_y, gyro_z=r.gyro_z,
            created_at=r.created_at  # 💡 [수정] first.created_at에서 r.created_at으로 변경!
        )
        for r in records
    ]
    
    # 3. 대용량 벌크 저장
    db.bulk_save_objects(db_records)
    db.commit()
    return len(db_records)
    