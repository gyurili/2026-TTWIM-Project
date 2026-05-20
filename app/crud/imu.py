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