import uuid
from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.imu import GaitBatchUpload, GaitBatchResponse
from app.crud import imu as crud
from app.services import imu as service

router = APIRouter(prefix="/imu", tags=["IMU"])


@router.post("/batch", response_model=GaitBatchResponse, status_code=201)
def upload_batch(
    payload: GaitBatchUpload,
    db: Session = Depends(get_db),
):
    """React Native에서 JSON 배열로 묶어서 전송"""
    session_id = str(uuid.uuid4())
    count = crud.bulk_create_gait_records(db, payload.records, session_id)
    return GaitBatchResponse(message="저장 완료", count=count)


@router.post("/upload-db", response_model=GaitBatchResponse, status_code=201)
async def upload_sqlite_db(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """React Native SQLite .db 파일을 그대로 전송받아 RDS에 저장"""
    session_id = str(uuid.uuid4())
    records = await service.parse_sqlite_upload(file)
    count = crud.bulk_create_gait_records(db, records, session_id)
    return GaitBatchResponse(message="저장 완료", count=count)