from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from src.workers.tasks import process_theft_analysis

app = FastAPI(title="GridGuard Production API")

class DetectionRequest(BaseModel):
    consumers: List[Dict]

@app.post("/detect", status_code=202)
async def detect_theft(request: DetectionRequest):
    """
    Submits a batch for analysis. Returns a job_id for polling.
    """
    # Offload the heavy work to Celery
    job = process_theft_analysis.delay(request.dict()['consumers'])
    return {"job_id": job.id, "status": "Processing"}

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """
    Poll this endpoint to retrieve the results once finished.
    """
    job_result = process_theft_analysis.AsyncResult(job_id)
    if job_result.ready():
        return {"status": "Completed", "result": job_result.result}
    return {"status": job_result.status}