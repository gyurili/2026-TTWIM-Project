from fastapi import APIRouter
from app.schemas.predict import PredictBatchRequest, PredictBatchResponse
from app.services import predict as service

router = APIRouter(tags=["Predict"])


@router.post("/predict", response_model=PredictBatchResponse)
def predict_heel_strike(data: PredictBatchRequest):
    return service.predict_batch(data)
