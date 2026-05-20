from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.imu import IMURecordCreate, IMUBatchUpload, IMUBatchResponse
from app.crud import imu as crud
from app.services import imu as service

router = APIRouter(prefix="/imu", tags=["IMU"])



# ── 2. JSON 배치 전송 ──────────────────────────────────────
@router.post("/batch", response_model=IMUBatchResponse, status_code=201)
def upload_batch(
    payload: IMUBatchUpload,
    db: Session = Depends(get_db),
):
    """React Native에서 JSON 배열로 묶어서 전송"""
    count = crud.bulk_create_imu_records(db, payload.records)
    return IMUBatchResponse(message="저장 완료", count=count)


# ── 3. SQLite .db 파일 직접 업로드 ───────────────────────────
@router.post("/upload-db", response_model=IMUBatchResponse, status_code=201)
async def upload_sqlite_db(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """React Native SQLite .db 파일을 그대로 전송받아 RDS에 저장"""
    records = await service.parse_sqlite_upload(file)
    count = crud.bulk_create_imu_records(db, records)
    return IMUBatchResponse(message="저장 완료", count=count)