from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import os
import json
from datetime import datetime

from app.services.chroma_service import chroma_service
from app.services.scraper_service import scraper_service
from app.services.ai_service import ai_service

router = APIRouter()

class SystemConfig(BaseModel):
    max_concurrent_requests: int = 10
    request_delay: float = 1.0
    max_crawl_depth: int = 3
    enable_ai_fallback: bool = True

@router.get("/system-status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
       
        chroma_stats = await chroma_service.get_collection_stats()
        
        ai_status = {
            'openrouter': bool(ai_service.openrouter_client),
            'voyage': bool(ai_service.voyage_client),
            'ollama': bool(ai_service.ollama_client)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'services': {
                'scraper': 'active',
                'chroma_db': 'active',
                'ai_services': ai_status
            },
            'statistics': {
                'documents_stored': chroma_stats['document_count'],
                'collection_name': chroma_stats['collection_name']
            },
            'system_health': 'healthy'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/configure")
async def configure_system(config: SystemConfig):
    """Configure system parameters"""
    try:
        
        scraper_service.max_concurrent_requests = config.max_concurrent_requests
        scraper_service.request_delay = config.request_delay
        
        return {
            'status': 'success',
            'message': 'System configuration updated',
            'config': config.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_system_logs(limit: int = 100):
    """Get system logs"""
    try:
        logs = [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'service': 'scraper',
                'message': 'System operational'
            }
        ]
        
        return {
            'logs': logs[-limit:],
            'total_logs': len(logs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backup")
async def create_backup():
    """Create system backup"""
    try:
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'chroma_stats': await chroma_service.get_collection_stats(),
            'system_config': {
                'version': '1.0.0',
                'services': ['scraper', 'rag', 'chroma']
            }
        }
        
        backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = os.path.join("./backups", backup_filename)
        
        os.makedirs("./backups", exist_ok=True)
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return {
            'status': 'success',
            'backup_file': backup_filename,
            'backup_path': backup_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_system_metrics():
    """Get detailed system metrics"""
    try:
        chroma_stats = await chroma_service.get_collection_stats()
        
        metrics = {
            'documents': {
                'total_count': chroma_stats['document_count'],
                'collection_name': chroma_stats['collection_name']
            },
            'system': {
                'uptime': 'N/A',  
                'memory_usage': 'N/A',  
                'cpu_usage': 'N/A' 
            },
            'services': {
                'scraper_active': True,
                'rag_active': True,
                'ai_services_active': True
            }
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
