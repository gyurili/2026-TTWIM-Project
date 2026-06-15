from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse
from app.services import predict as service

router = APIRouter(tags=["Predict"])


@router.post("/predict", response_model=PredictBatchResponse)
def predict_heel_strike(data: PredictBatchRequest):
    try:
        return service.predict_batch(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
