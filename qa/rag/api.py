from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crew import run_crew
import uvicorn

from common.settings import settings

app = FastAPI()

class UserRequest(BaseModel):
    message: str

@app.post("/process")
async def process_request(request: UserRequest):
    try:
        response = run_crew(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(settings.RAG_CREWAI_URL.split(":")[-1].split("/")[0]
    )
)