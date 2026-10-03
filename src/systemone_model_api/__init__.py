import uvicorn


def main() -> None:
    uvicorn.run("systemone_model_api.api:app", host="0.0.0.0", port=8000)
