import logging
from typing import Any

import laya
from fastapi import Depends, FastAPI, HTTPException, status

from laya_api.auth import authenticate
from laya_api.models import PredictRequest

logger = logging.getLogger("uvicorn.error")

# Loaded exactly once per worker.
agent = laya.load("convaiinnovations/laya")

app = FastAPI(
    title="Laya API",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    dependencies=[Depends(authenticate)],
)
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        # exclude_none: a noul question without criteria must not reach Laya
        # as {"criteria": None}; the key has to be absent entirely.
        result = agent.predict(
            request.state,
            {
                question_id: question.model_dump(exclude_none=True)
                for question_id, question in request.questions.items()
            },
        )
    except Exception as exc:
        # Log the traceback server-side; keep model internals out of the response.
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc

    return result
