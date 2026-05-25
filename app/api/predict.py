from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictRequest, PredictResponse
from app.services import predict as service

router = APIRouter(tags=["Predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_heel_strike(data: PredictRequest):
    try:
        return service.predict(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
