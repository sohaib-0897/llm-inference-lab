from fastapi import APIRouter

from backend.app.schemas import SystemInfoResponse
from backend.app.services.runtime_service import runtime_info

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system", response_model=SystemInfoResponse)
def system() -> SystemInfoResponse:
    return SystemInfoResponse(**runtime_info())
