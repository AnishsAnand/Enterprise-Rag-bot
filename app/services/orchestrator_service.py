# orchestrator_service.py
import asyncio, uuid, logging
from typing import Dict, Any, Optional
from datetime import datetime
from app.services.ai_service import ai_service
from app.services.postgres_service import postgres_service
from app.services.scraper_service import scraper_service
from app.services.file_service import file_service
from app.services.image_service import image_service
from app.services.artifact_store import artifact_store

logger = logging.getLogger(__name__)

class OrchestratorService:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    async def handle_user_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for user queries (automation or chat)."""
        task_id = str(uuid.uuid4())
        intent = await ai_service.detect_intent(query)
        logger.info(f"[{task_id}] Detected intent: {intent}")

        if intent == "generate_pdf":
            return await self._handle_pdf_generation(task_id, query)
        elif intent == "generate_image":
            return await self._handle_image_generation(task_id, query)
        elif intent == "scrape_url":
            return await self._handle_scraping(task_id, query)
        else:
            answer = await ai_service.generate_text_response(query)
            return {"type": "chat", "task_id": task_id, "answer": answer, "status": "completed"}  

    async def _handle_pdf_generation(self, task_id: str, query: str) -> Dict[str, Any]:
        """Generate a PDF document based on RAG knowledge."""
        results = await postgres_service.search_documents(query, n_results=10)
        context = [r["content"] for r in results if r.get("content")]
        ai_summary = await ai_service.generate_summary("\n".join(context))
        file_bytes = await file_service.generate_pdf(ai_summary, title=f"Report - {query}")
        artifact = artifact_store.save(file_bytes, mime="application/pdf", filename=f"report_{task_id}.pdf")

        return {
            "type": "pdf_generation",
            "task_id": task_id,
            "status": "completed",
            "message": f"✅ Report generated for: {query}",
            "artifact_url": artifact["url"],
        }

    async def _handle_image_generation(self, task_id: str, query: str) -> Dict[str, Any]:
        """Use AI service to create an image based on user request."""
        img_bytes = await image_service.generate_image(query)
        artifact = artifact_store.save(img_bytes, mime="image/png", filename=f"image_{task_id}.png")
        return {
            "type": "image_generation",
            "task_id": task_id,
            "status": "completed",
            "message": f"🖼️ Image created for: {query}",
            "artifact_url": artifact["url"],
        }

    async def _handle_scraping(self, task_id: str, query: str) -> Dict[str, Any]:
        """Scrape a URL mentioned in user query."""
        import re
        urls = re.findall(r"https?://[^\s]+", query)
        if not urls:
            return {"type": "scrape", "status": "need_url", "message": "Please provide a valid URL to scrape."}

        url = urls[0]
        result = await scraper_service.scrape_url(url, {"extract_text": True, "extract_images": True})
        stored = await postgres_service.store_documents(result["content"]["rag_documents"])
        return {
            "type": "scrape",
            "task_id": task_id,
            "status": "completed",
            "message": f"✅ Scraped and stored content from: {url}",
            "stored_count": stored,
        }

orchestrator_service = OrchestratorService()
