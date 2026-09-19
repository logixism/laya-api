import uvicorn


def main() -> None:
    uvicorn.run("laya_api.api:app", host="0.0.0.0", port=8000)
