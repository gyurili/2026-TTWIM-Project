from fastapi import APIRouter, HTTPException, status

from app.schemas.footdrag import (
    FootDragBatchRequest,
    FootDragBatchResponse,
    FootDragPredictionResponse,
    FootDragStepRequest,
)
from app.services import footdrag_predict as service
from app.services.footdrag_predict import ModelNotLoadedError

router = APIRouter(prefix="/api/v2/footdrag", tags=["Foot Drag"])


@router.post("/predict", response_model=FootDragPredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(data: FootDragStepRequest):
    try:
        return service.predict_step(data)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/batch", response_model=FootDragBatchResponse, status_code=status.HTTP_200_OK)
def predict_batch(data: FootDragBatchRequest):
    try:
        return service.predict_batch(data)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
