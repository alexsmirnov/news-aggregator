from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from news.scheduler import NEWS_DIGEST_JOB_ID

router = APIRouter()


class AggregateStatus(BaseModel):
    job_id: str
    scheduler_running: bool
    job_pending: bool
    next_run_time: datetime | None


def _aggregate_status(request: Request) -> AggregateStatus:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler is unavailable",
        )

    job = scheduler.get_job(NEWS_DIGEST_JOB_ID)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aggregate job is unavailable",
        )

    return AggregateStatus(
        job_id=job.id,
        scheduler_running=scheduler.running,
        job_pending=job.pending,
        next_run_time=job.next_run_time,
    )


@router.get("/aggregate")
async def get_aggregate_status(request: Request) -> AggregateStatus:
    return _aggregate_status(request)


@router.post("/aggregate", status_code=status.HTTP_202_ACCEPTED)
async def trigger_aggregate(request: Request) -> AggregateStatus:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None or not scheduler.running:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler is unavailable",
        )

    job = scheduler.get_job(NEWS_DIGEST_JOB_ID)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aggregate job is unavailable",
        )

    job.modify(next_run_time=datetime.now(UTC))
    return _aggregate_status(request)
