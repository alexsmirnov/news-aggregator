import uvicorn

def main() -> None:

    uvicorn.run("news.main:app", host="0.0.0.0", port=4090, reload=True)
