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

    # 💡 [수정] 이제 모든 레코드 객체에 직접 session_id 루프 돌며 주입할 필요가 없어졌습니다!
    # 대신 crud 함수에 session_id를 직접 인자로 넘깁니다.
    count = crud.bulk_create_imu_records(db, payload.records, session_id)
    return IMUBatchResponse(message="저장 완료", count=count)


# ── SQLite .db 파일 직접 업로드 ───────────────────────────────
@router.post("/upload-db", response_model=IMUBatchResponse, status_code=201)
async def upload_sqlite_db(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """React Native SQLite .db 파일을 그대로 전송받아 RDS에 저장"""
    session_id = str(uuid.uuid4())  # 💡 [수정] .db 파일 업로드 시에도 고유 세션 ID 발급
    records = await service.parse_sqlite_upload(file)
    
    # 💡 [수정] 인자로 생성한 session_id를 넘겨줍니다.
    count = crud.bulk_create_imu_records(db, records, session_id)
    return IMUBatchResponse(message="저장 완료", count=count)