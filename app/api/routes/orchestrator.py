from fastapi import APIRouter, HTTPException
from app.services.orchestrator_service import orchestrator_service

router = APIRouter(prefix="/api/automation", tags=["Automation"])

@router.post("/execute")
async def execute_task(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    return await orchestrator_service.handle_user_query(query)
