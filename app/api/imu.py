import uuid
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.imu import IMURecordCreate, IMUBatchUpload, IMUBatchResponse
from app.crud import imu as crud
from app.services import imu as service

router = APIRouter(prefix="/imu", tags=["IMU"])


# ── 배치 전송 ─────────────────────────────────────────────────
@router.post("/batch", response_model=IMUBatchResponse, status_code=201)
def upload_batch(
    payload: IMUBatchUpload,
    db: Session = Depends(get_db),
):
    """React Native에서 JSON 배열로 묶어서 전송"""
    session_id = str(uuid.uuid4())  # 자동 생성

    for record in payload.records:
        record.session_id = session_id  # 모든 레코드에 같은 session_id 적용

    count = crud.bulk_create_imu_records(db, payload.records)
    return IMUBatchResponse(message="저장 완료", count=count)


# ── SQLite .db 파일 직접 업로드 ───────────────────────────────
@router.post("/upload-db", response_model=IMUBatchResponse, status_code=201)
async def upload_sqlite_db(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """React Native SQLite .db 파일을 그대로 전송받아 RDS에 저장"""
    records = await service.parse_sqlite_upload(file)
    count = crud.bulk_create_imu_records(db, records)
    return IMUBatchResponse(message="저장 완료", count=count)