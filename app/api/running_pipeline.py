from fastapi import APIRouter, HTTPException
from app.schemas.running_pipeline import RunningPipelineBatchRequest, RunningPipelineBatchResponse
from app.services import running_pipeline as service

router = APIRouter(tags=["RunningPipeline"])


@router.post("/running-pipeline/predict", response_model=RunningPipelineBatchResponse)
def predict_running_pipeline(data: RunningPipelineBatchRequest):
    try:
        return service.predict_batch(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
