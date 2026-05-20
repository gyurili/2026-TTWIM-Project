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
            # 💡 [보안] .db 파일 파싱할 때는 회원/신체 정보가 없으므로 None으로 명시적 채움 (스키마 에러 방지)
            user_id=None,
            session_id=None,
            label=None,
            weight=None,
            height=None
        )
        for row in rows
    ]

import re  # 💡 파일 최상단에 import re 가 없다면 추가해 주세요!
from datetime import datetime

def _parse_korean_datetime(dt_str: str) -> datetime:
    """
    "2026. 5. 20. 오후 9:36:44" 또는 "2026. 5. 20. 오후 9:36:45.8" 
    한국어 날짜 형식을 자릿수 오류 없이 완벽하게 24시간제 datetime으로 변환합니다.
    """
    dt_str = dt_str.strip()
    
    # 1. 정규식으로 숫자들만 쏙쏙 뽑아냅니다.
    # 결과 예시: ['2026', '5', '20', '9', '36', '44', '8']
    numbers = re.findall(r'\d+', dt_str)
    
    # 만약 소수점(밀리초)이 없다면 뒤에 0을 채워줍니다.
    if len(numbers) == 6:
        numbers.append("0")
        
    year, month, day, hour, minute, second, micro = map(int, numbers)
    
    # 2. '오후' 변환 처리 (12시간제 -> 24시간제)
    # 12시는 오후 12시(24시간제 12시), 오전 12시(24시간제 0시) 예외처리 포함
    if "오후" in dt_str and hour != 12:
        hour += 12
    elif "오전" in dt_str and hour == 12:
        hour = 0
        
    # 밀리초 자릿수 맞추기 (예: 8 -> 800000 마이크로초)
    micro_str = numbers[6].ljust(6, '0')[:6]
    micro = int(micro_str)

    # 3. 안전하게 정수값들로 datetime 객체 조립
    return datetime(year, month, day, hour, minute, second, micro)