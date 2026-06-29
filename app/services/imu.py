import json
import sqlite3
import tempfile
import os
from datetime import datetime
from typing import List
from fastapi import UploadFile
from app.core.exceptions import UploadValidationError
from app.schemas.imu import GaitRecordCreate


async def parse_sqlite_upload(file: UploadFile) -> List[GaitRecordCreate]:
    if not file.filename.endswith(".db"):
        raise UploadValidationError(".db 파일만 업로드 가능합니다")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        records = _read_sqlite(tmp_path)
    finally:
        os.unlink(tmp_path)

    return records


def _read_sqlite(db_path: str) -> List[GaitRecordCreate]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT windowIndex, startTime, endTime, label, subject,
                rawAccX, rawAccY, rawAccZ,
                rawGyroX, rawGyroY, rawGyroZ
        FROM sensor_data
        ORDER BY id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    return [
        GaitRecordCreate(
            window_index = row["windowIndex"],
            start_time   = datetime.fromisoformat(row["startTime"]),
            end_time     = datetime.fromisoformat(row["endTime"]),
            label        = row["label"] or "normal",
            subject      = row["subject"] or "",
            raw_acc_x    = json.loads(row["rawAccX"]),
            raw_acc_y    = json.loads(row["rawAccY"]),
            raw_acc_z    = json.loads(row["rawAccZ"]),
            raw_gyro_x   = json.loads(row["rawGyroX"]),
            raw_gyro_y   = json.loads(row["rawGyroY"]),
            raw_gyro_z   = json.loads(row["rawGyroZ"]),
        )
        for row in rows
    ]
