import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from systemone_model_api.auth import authenticate
from systemone_model_api.backends import load_predictor
from systemone_model_api.config import API_KEY, DEVICE, MODEL_NAME
from systemone_model_api.media import decode_images
from systemone_model_api.models import PredictRequest

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("SYSTEMONE_API_KEY is not set")
    # One resident model per worker; importing the API never downloads weights.
    app.state.predictor = load_predictor(MODEL_NAME, DEVICE)
    try:
        yield
    finally:
        del app.state.predictor

app = FastAPI(
    title="SystemOne Model API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    dependencies=[Depends(authenticate)],
)
def predict(request: PredictRequest) -> dict[str, Any]:
    if request.model is not None and request.model != MODEL_NAME:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"This server is configured for {MODEL_NAME}",
        )
    if request.images and MODEL_NAME == "laya":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Images require clef or clef-flash",
        )
    try:
        images = decode_images(request.images)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        result = app.state.predictor(
            request.state,
            {
                question_id: question.model_dump(exclude_none=True)
                for question_id, question in request.questions.items()
            },
            images,
        )
    except Exception as exc:
        # Log the traceback server-side; keep model internals out of the response.
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc
    finally:
        for image in images:
            image.close()

    return result
