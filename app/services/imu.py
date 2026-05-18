import sqlite3
import tempfile
import os
from datetime import datetime
from typing import List
from fastapi import UploadFile, HTTPException
from app.schemas.imu import IMURecordCreate


async def parse_sqlite_upload(file: UploadFile) -> List[IMURecordCreate]:
    """
    React Native SQLite .db 파일을 받아서
    IMURecordCreate 리스트로 변환
    """
    if not file.filename.endswith(".db"):
        raise HTTPException(status_code=400, detail=".db 파일만 업로드 가능합니다")

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        records = _read_sqlite(tmp_path)
    finally:
        os.unlink(tmp_path)  # 임시 파일 항상 삭제

    return records


def _read_sqlite(db_path: str) -> List[IMURecordCreate]:
    """SQLite sensor_data 테이블 → Pydantic 스키마 변환"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT side,
               AccX, AccY, AccZ,
               GyroX, GyroY, GyroZ,
               createdAt
        FROM sensor_data
    """)
    rows = cur.fetchall()
    conn.close()

    return [
        IMURecordCreate(
            side=row["side"],
            acc_x=row["AccX"],
            acc_y=row["AccY"],
            acc_z=row["AccZ"],
            gyro_x=row["GyroX"],
            gyro_y=row["GyroY"],
            gyro_z=row["GyroZ"],
            created_at=_parse_korean_datetime(row["createdAt"]),
        )
        for row in rows
    ]


def _parse_korean_datetime(dt_str: str) -> datetime:
    """
    "2025. 5. 18. 오전 10:00:00" → datetime
    toLocaleString('ko-KR') 형식 파싱
    """
    dt_str = dt_str.strip()
    dt_str = dt_str.replace("오전", "AM").replace("오후", "PM")
    return datetime.strptime(dt_str, "%Y. %m. %d. %p %I:%M:%S")