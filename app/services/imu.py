import json
import sqlite3
import tempfile
import os
from datetime import datetime
from typing import List
from fastapi import UploadFile, HTTPException
from app.schemas.imu import GaitRecordCreate


async def parse_sqlite_upload(file: UploadFile) -> List[GaitRecordCreate]:
    if not file.filename.endswith(".db"):
        raise HTTPException(status_code=400, detail=".db 파일만 업로드 가능합니다")

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
        SELECT side, stepIndex, startTime, endTime,
               peakAccZ, cadence, contactTime, label,
               resampledAccX, resampledAccY, resampledAccZ,
               resampledGyroX, resampledGyroY, resampledGyroZ
        FROM sensor_data
        ORDER BY id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    return [
        GaitRecordCreate(
            side             = row["side"],
            step_index       = row["stepIndex"],
            start_time       = datetime.fromisoformat(row["startTime"]),
            end_time         = datetime.fromisoformat(row["endTime"]),
            peak_acc_z       = row["peakAccZ"],
            cadence          = row["cadence"],
            contact_time     = row["contactTime"],
            label            = row["label"] or "normal",
            resampled_acc_x  = json.loads(row["resampledAccX"]),
            resampled_acc_y  = json.loads(row["resampledAccY"]),
            resampled_acc_z  = json.loads(row["resampledAccZ"]),
            resampled_gyro_x = json.loads(row["resampledGyroX"]),
            resampled_gyro_y = json.loads(row["resampledGyroY"]),
            resampled_gyro_z = json.loads(row["resampledGyroZ"]),
        )
        for row in rows
    ]