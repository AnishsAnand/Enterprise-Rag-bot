from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import mimetypes
from io import BytesIO, StringIO
import csv
from PyPDF2 import PdfReader
from docx import Document
import openpyxl
from bs4 import BeautifulSoup
import os
import logging
import re
import urllib.parse
import difflib
import inspect

from app.services.scraper_service import scraper_service
from app.services.milvus_service import milvus_service  # CHANGED: from chroma_service to milvus_service
from app.services.ai_service import ai_service

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------- Request models ----------
class WidgetQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The search query")
    max_results: int = Field(default=50, ge=1, le=100, description="Maximum number of results")
    include_sources: bool = Field(default=True, description="Include source information")
    enable_advanced_search: bool = Field(default=True, description="Enable advanced search features")
    search_depth: str = Field(default="balanced", pattern="^(quick|balanced|deep)$")

class WidgetScrapeRequest(BaseModel):
    url: HttpUrl
    store_in_knowledge: bool = True
    extract_images: bool = True
    wait_for_js: bool = False

class BulkScrapeRequest(BaseModel):
    base_url: HttpUrl
    max_depth: int = Field(default=2, ge=1, le=5)
    max_urls: int = Field(default=50, ge=1, le=500)
    auto_store: bool = True
    domain_filter: Optional[str] = None

# ---------- Utilities ----------
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Chunk text with sentence/paragraph boundaries"""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + (chunk_size // 2):
                end = paragraph_break
            else:
                sentence_break = text.rfind(". ", start, end)
                if sentence_break > start + (chunk_size // 2):
                    end = sentence_break + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # compute next start with overlap but ensure forward progress
        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            start = next_start
    return chunks

def _enhanced_similarity(query: str, text: str) -> float:
    """Return [0,1] similarity score"""
    if not query or not text:
        return 0.0
    query_norm = " ".join(query.lower().split())
    text_norm = " ".join(text.lower().split())
    seq_ratio = difflib.SequenceMatcher(None, query_norm, text_norm).ratio()
    query_words, text_words = set(query_norm.split()), set(text_norm.split())
    overlap = len(query_words & text_words) / len(query_words | text_words) if query_words else 0.0
    substring_bonus = 0.2 if query_norm in text_norm else 0.0
    score = 0.4 * seq_ratio + 0.4 * overlap + 0.2 * substring_bonus
    return min(1.0, max(0.0, score))

def _extract_key_concepts(text: str) -> List[str]:
    """Extract key terms from text"""
    if not text:
        return []
    patterns = [r"\b[A-Z]{2,}\b", r"\b\w+(?:_\w+)+\b", r"\b\w+(?:-\w+)+\b", r"\b\d+(?:\.\d+)?\w*\b"]
    concepts = []
    for p in patterns:
        concepts.extend(re.findall(p, text))
    words = re.findall(r"\b[a-zA-Z]{5,}\b", text)
    stopwords = {"about", "after", "again", "before", "being", "could", "while", "would"}
    meaningful = [w.lower() for w in words if w.lower() not in stopwords]
    combined = list(dict.fromkeys(concepts + meaningful))  # preserve order and dedupe
    return combined[:15]

# helper to call methods that might be sync or async
async def call_maybe_async(fn, *args, **kwargs):
    """
    Call a function that might be sync or async. If it returns an awaitable, await it.
    This allows service objects (milvus_service, ai_service) to expose sync or async methods.
    """
    if not callable(fn):
        raise RuntimeError("Provided object is not callable")
    try:
        result = fn(*args, **kwargs)
    except TypeError:
        # sometimes partial / bound methods require different invocation
        result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result

# ---------- Endpoints ----------
@router.post("/widget/query")
async def widget_query(request: WidgetQueryRequest):
    """Enhanced query processing with improved search and relevance"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing widget query: '{query}' (depth: {request.search_depth})")

        search_params = {
            "quick": {"max_results": min(request.max_results, 30), "use_reranking": False},
            "balanced": {"max_results": request.max_results, "use_reranking": True},
            "deep": {"max_results": min(request.max_results * 2, 100), "use_reranking": True},
        }
        search_config = search_params.get(request.search_depth, search_params["balanced"])

        # search_documents may be sync or async; use call_maybe_async
        try:
            search_results = await call_maybe_async(
                getattr(milvus_service, "search_documents", milvus_service),
                query,
                n_results=search_config["max_results"]
            )
        except Exception as e:
            logger.exception(f"Error while searching documents: {e}")
            search_results = []

        # If no search results, try LLM-only fallback (so widget doesn't always say "no info")
        if not search_results:
            logger.warning("No search results found; attempting LLM-only fallback.")
            try:
                answer = await call_maybe_async(ai_service.generate_response, query, [])
            except Exception as e:
                logger.warning(f"LLM fallback failed: {e}")
                answer = None

            if answer:
                summary = None
                try:
                    summary = await call_maybe_async(ai_service.generate_summary, answer, max_sentences=3, max_chars=600)
                except Exception:
                    summary = (answer[:600] + "...") if len(answer) > 600 else answer

                return {
                    "query": query,
                    "answer": answer,
                    "expanded_context": None,
                    "step_count": 0,
                    "steps": [],
                    "images": [],
                    "sources": [],
                    "has_sources": False,
                    "confidence": 0.45,
                    "search_depth": request.search_depth,
                    "results_found": 0,
                    "results_used": 0,
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary or "No summary available."
                }

            # final fallback if LLM also fails
            return {
                "query": query,
                "answer": "I don't have any relevant information in my knowledge base to answer your question. Please try rephrasing your query or add more context.",
                "steps": [],
                "images": [],
                "sources": [],
                "has_sources": False,
                "confidence": 0.0,
                "search_depth": request.search_depth,
                "timestamp": datetime.now().isoformat()
            }

        # Advanced filtering & reranking
        if request.enable_advanced_search:
            avg_score = sum(r.get("relevance_score", 0) for r in search_results) / max(1, len(search_results))
            min_threshold = max(0.3, avg_score * 0.6)
            filtered_results = [r for r in search_results if r.get("relevance_score", 0) >= min_threshold]
            if len(filtered_results) < max(3, int(len(search_results) * 0.2)):
                filtered_results = search_results[:max(5, request.max_results // 2)]
        else:
            filtered_results = search_results[: request.max_results]

        # Build base context to send to LLM
        base_context = []
        for result in filtered_results:
            content = result.get("content", "") if isinstance(result, dict) else ""
            if content and len(content.strip()) > 50:
                score = result.get("relevance_score", 0.5)
                if score > 0.7:
                    base_context.append(content)
                elif score > 0.5:
                    base_context.append(content[:1500])
                else:
                    base_context.append(content[:800])

        if not base_context:
            base_context = [r.get("content", "")[:1000] for r in filtered_results[:3]]

        # Ask ai_service for an enhanced response
        try:
            enhanced_result = await call_maybe_async(ai_service.generate_enhanced_response, query, base_context, None)
            answer = (enhanced_result or {}).get("text", "") if isinstance(enhanced_result, dict) else (enhanced_result or "")
            expanded_context = (enhanced_result or {}).get("expanded_context", "") if isinstance(enhanced_result, dict) else ""
            confidence = (enhanced_result or {}).get("quality_score", 0.0) if isinstance(enhanced_result, dict) else 0.0
        except Exception as e:
            logger.warning(f"Enhanced response generation failed: {e}")
            try:
                answer = await call_maybe_async(ai_service.generate_response, query, base_context[:3])
            except Exception as e2:
                logger.error(f"Fallback generate_response also failed: {e2}")
                answer = ""
            expanded_context = "\n\n".join(base_context[:2]) if base_context else ""
            confidence = 0.6

        # Generate stepwise response (may be sync or async)
        working_context = [expanded_context] if expanded_context else base_context[:3]
        try:
            steps_data = await call_maybe_async(ai_service.generate_stepwise_response, query, working_context)
        except Exception as e:
            logger.warning(f"Stepwise generation failed: {e}")
            steps_data = []

        if not steps_data:
            if answer:
                sentences = [s.strip() for s in answer.split(".") if s.strip()]
                steps_data = [{"text": (s + "."), "type": "info"} for s in sentences[:5]]
            else:
                steps_data = [{"text": "Unable to generate structured response.", "type": "info"}]

        # Select images from metadata in filtered_results
        candidate_images = []
        query_concepts = set(_extract_key_concepts(query.lower()))
        answer_concepts = set(_extract_key_concepts(answer.lower())) if answer else set()
        all_concepts = query_concepts | answer_concepts

        for result in filtered_results:
            meta = result.get("metadata", {}) or {}
            page_url = meta.get("url", "")
            page_title = meta.get("title", "")
            relevance_score = result.get("relevance_score", 0.0)

            images = meta.get("images", []) if isinstance(meta.get("images", []), list) else []
            for img in images:
                if not isinstance(img, dict) or not img.get("url"):
                    continue
                # noise filtering
                u = img.get("url", "").lower()
                if any(noise in u for noise in ["logo", "icon", "favicon", "sprite", "banner"]):
                    continue

                img_text = (img.get("text", "") or "").lower()
                img_alt = (img.get("alt", "") or "").lower()
                img_type = (img.get("type", "") or "").lower()

                img_concepts = set(_extract_key_concepts(img_text))
                concept_overlap = len(all_concepts & img_concepts) if all_concepts and img_concepts else 0
                text_similarity = _enhanced_similarity(query, img_text)

                type_bonus = 0.2 if img_type in ["diagram", "chart", "screenshot", "illustration"] else 0.0

                image_score = (
                    text_similarity * 0.4
                    + (concept_overlap / max(len(all_concepts), 1)) * 0.3
                    + relevance_score * 0.2
                    + type_bonus
                )

                if image_score > 0.15:
                    candidate_images.append(
                        {
                            "url": img.get("url"),
                            "alt": img.get("alt", ""),
                            "type": img.get("type", ""),
                            "caption": img.get("caption", ""),
                            "source_url": page_url,
                            "source_title": page_title,
                            "relevance_score": round(image_score, 3),
                            "text": img_text[:500],
                        }
                    )

        # Deduplicate and keep top
        seen_urls = set()
        unique_images = []
        for img in sorted(candidate_images, key=lambda x: x["relevance_score"], reverse=True):
            url = img.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_images.append(img)
        selected_images = unique_images[:12]

        # Debug log for filtered results & initial selected images
        logger.debug("Filtered_results count: %d, top relevance scores: %s",
                     len(filtered_results),
                     [round(r.get("relevance_score", 0), 3) for r in filtered_results[:5]])
        logger.debug("Selected images (initial): %d", len(selected_images))

        # If no images selected via scoring, do a relaxed fallback: pick first non-noise images from metadata
        if not selected_images:
            logger.debug("No images passed scoring threshold - attempting relaxed metadata fallback.")
            relaxed = []
            for result in filtered_results:
                meta = result.get("metadata", {}) or {}
                page_url = meta.get("url", "")
                page_title = meta.get("title", "")
                images = meta.get("images", []) if isinstance(meta.get("images", []), list) else []
                for img in images:
                    if not isinstance(img, dict) or not img.get("url"):
                        continue
                    u = img.get("url", "").lower()
                    if any(noise in u for noise in ["logo", "icon", "favicon", "sprite", "banner"]):
                        continue
                    url = img.get("url")
                    if url in seen_urls:
                        continue
                    relaxed.append(
                        {
                            "url": url,
                            "alt": img.get("alt", ""),
                            "type": img.get("type", ""),
                            "caption": img.get("caption", ""),
                            "source_url": page_url,
                            "source_title": page_title,
                            "relevance_score": round(result.get("relevance_score", 0.0), 3),
                            "text": (img.get("text", "") or "")[:500],
                        }
                    )
                    seen_urls.add(url)
                    if len(relaxed) >= 12:
                        break
                if relaxed:
                    break
            if relaxed:
                logger.info(f"Relaxed image fallback added {len(relaxed)} images.")
            selected_images = relaxed[:12]

        # If step-level images were returned by the step generator, prefer/merge URL images into selected_images.
        # Note: image_prompt (text descriptions) are handled later and are not inserted into selected_images.
        step_level_images_added = 0
        for step in steps_data:
            if isinstance(step, dict):
                # Prefer explicit dict in step["image"] first
                possible_img = step.get("image") or step.get("image_url") or step.get("image_data")
                # If the LLM returned a dict with 'url', insert it to front of selected_images
                if isinstance(possible_img, dict) and possible_img.get("url"):
                    url = possible_img.get("url")
                    if url not in seen_urls:
                        img_obj = {
                            "url": url,
                            "alt": possible_img.get("alt", "") or step.get("alt", ""),
                            "type": possible_img.get("type", "") or "",
                            "caption": possible_img.get("caption", "") or step.get("caption", ""),
                            "source_url": possible_img.get("source_url", "") or "",
                            "source_title": possible_img.get("source_title", "") or "",
                            "relevance_score": round(possible_img.get("relevance_score", 0.0), 3),
                            "text": (possible_img.get("text", "") or "")[:500],
                        }
                        selected_images.insert(0, img_obj)
                        seen_urls.add(url)
                        step_level_images_added += 1
                # If LLM returned a bare URL string via step['image'] or step['image_url'], add it
                elif isinstance(step.get("image"), str) and step.get("image").startswith("http"):
                    url = step.get("image")
                    if url not in seen_urls:
                        img_obj = {
                            "url": url,
                            "alt": step.get("alt", ""),
                            "type": "",
                            "caption": step.get("caption", ""),
                            "source_url": "",
                            "source_title": "",
                            "relevance_score": 0.0,
                            "text": "",
                        }
                        selected_images.insert(0, img_obj)
                        seen_urls.add(url)
                        step_level_images_added += 1

        if step_level_images_added:
            selected_images = selected_images[:12]
            logger.debug(f"Added {step_level_images_added} step-level URL images to selected_images.")

        # Build sources list
        sources = []
        if request.include_sources:
            for result in filtered_results:
                meta = result.get("metadata", {}) or {}
                content_preview_raw = result.get("content", "") or ""
                content_preview = (
                    content_preview_raw[:300] + "..." if len(content_preview_raw) > 300 else content_preview_raw
                )
                sources.append(
                    {
                        "url": meta.get("url", ""),
                        "title": meta.get("title", "Untitled"),
                        "relevance_score": round(result.get("relevance_score", 0), 3),
                        "content_preview": content_preview,
                        "domain": meta.get("domain", ""),
                        "last_updated": meta.get("timestamp", ""),
                    }
                )

        # Summary
        summary_input = answer if answer else expanded_context
        if summary_input:
            try:
                summary = await call_maybe_async(ai_service.generate_summary, summary_input, max_sentences=4, max_chars=600)
            except Exception:
                summary = summary_input[:600] + "..." if len(summary_input) > 600 else summary_input
        else:
            summary = "No summary available."

        # Combine steps with images (respect step-level image urls and image_prompts)
        steps_with_images = []
        for i, step in enumerate(steps_data):
            step_obj = {"index": i + 1, "text": step.get("text", ""), "type": step.get("type", "action")}
            assigned_img = None

            if isinstance(step, dict):
                # 1) If step contains an explicit image dict with URL, use it
                si = step.get("image") if isinstance(step.get("image"), (dict, str)) else None
                if isinstance(si, dict) and si.get("url"):
                    assigned_img = {
                        "url": si.get("url"),
                        "alt": si.get("alt", "") or step.get("alt", "") or "",
                        "caption": si.get("caption", "") or step.get("caption", "") or "",
                        "relevance_score": si.get("relevance_score", None),
                    }
                elif isinstance(si, str) and si.startswith("http"):
                    assigned_img = {"url": si, "alt": step.get("alt", "") or "", "caption": step.get("caption", "") or "", "relevance_score": None}

                # 2) If step provides an image_prompt (string), attach it (prefer over fallback images)
                # Accept both step["image_prompt"] and step["image"] containing "image_prompt"
                if not assigned_img:
                    image_prompt = None
                    if isinstance(step.get("image_prompt"), str) and step.get("image_prompt").strip():
                        image_prompt = step.get("image_prompt").strip()
                    elif isinstance(step.get("image"), dict) and isinstance(step["image"].get("image_prompt"), str):
                        image_prompt = step["image"].get("image_prompt").strip()
                    elif isinstance(step.get("image"), str) and not step.get("image").startswith("http") and len(step.get("image").strip()) > 0:
                        # some LLMs may return a plain prompt string in step["image"]
                        image_prompt = step.get("image").strip()

                    if image_prompt:
                        # store the prompt so downstream clients can generate/display the image
                        assigned_img = {"image_prompt": image_prompt}

            # 3) Fallback: use selected_images by position if no step-level image/prompt present
            if not assigned_img and selected_images and i < len(selected_images):
                step_img = selected_images[i]
                # ensure we have a URL
                if step_img.get("url"):
                    assigned_img = {
                        "url": step_img.get("url"),
                        "alt": step_img.get("alt", "") or "",
                        "caption": step_img.get("caption", "") or "",
                        "relevance_score": step_img.get("relevance_score"),
                    }

            # Attach assigned image object (could be a URL dict or image_prompt dict)
            if assigned_img:
                step_obj["image"] = assigned_img

            steps_with_images.append(step_obj)

        final_confidence = min(
            1.0,
            (
                (confidence or 0.0) * 0.4
                + (len(filtered_results) / max(request.max_results, 10)) * 0.3
                + (1.0 if answer and len(answer) > 100 else 0.5) * 0.3
            ),
        )

        # Debug summary of final images & steps
        logger.debug("Final selected_images: %d, steps_with_images: %d", len(selected_images), len(steps_with_images))

        return {
            "query": query,
            "answer": answer or "I was unable to generate a comprehensive answer based on the available information.",
            "expanded_context": expanded_context if request.enable_advanced_search else None,
            "step_count": len(steps_with_images),
            "steps": steps_with_images,
            "images": selected_images,
            "sources": sources,
            "has_sources": len(sources) > 0,
            "confidence": round(final_confidence, 3),
            "search_depth": request.search_depth,
            "results_found": len(search_results),
            "results_used": len(filtered_results),
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Widget query error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/widget/scrape")
async def widget_scrape(request: WidgetScrapeRequest, background_tasks: BackgroundTasks):
    """Enhanced scraping with better error handling and options"""
    try:
        logger.info(f"Scraping URL: {request.url}")
        scrape_params = {
            "extract_text": True,
            "extract_links": False,
            "extract_images": request.extract_images,
            "extract_tables": True,
            "scroll_page": request.wait_for_js,
            "wait_for_element": "body" if request.wait_for_js else None,
            "output_format": "json",
        }

        result = await call_maybe_async(scraper_service.scrape_url, str(request.url), scrape_params)
        if not result or result.get("status") != "success":
            raise HTTPException(status_code=400, detail=f"Scraping failed: {result.get('error', 'Unknown error') if result else 'No result'}")

        content = result.get("content", {}) or {}
        page_text = content.get("text", "") or ""
        if len(page_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Scraped content is too short or empty")

        if request.store_in_knowledge:
            if len(page_text) > 2000:
                chunks = chunk_text(page_text, chunk_size=1500, overlap=200)
                documents_to_store = []
                for i, chunk in enumerate(chunks):
                    documents_to_store.append(
                        {
                            "content": chunk,
                            "url": f"{str(request.url)}#chunk-{i}",
                            "title": content.get("title", "") or f"Content from {request.url}",
                            "format": "text/html",
                            "timestamp": datetime.now().isoformat(),
                            "source": "widget_scrape",
                            "images": content.get("images", []) if i == 0 else [],
                        }
                    )
            else:
                documents_to_store = [
                    {
                        "content": page_text,
                        "url": str(request.url),
                        "title": content.get("title", "") or f"Content from {request.url}",
                        "format": "text/html",
                        "timestamp": datetime.now().isoformat(),
                        "source": "widget_scrape",
                        "images": content.get("images", []) or [],
                    }
                ]

            background_tasks.add_task(store_document_task, documents_to_store)

        try:
            summary = await call_maybe_async(ai_service.generate_summary, page_text, max_sentences=4, max_chars=800)
        except Exception:
            summary = page_text[:800] + "..." if len(page_text) > 800 else page_text

        return {
            "status": "success",
            "url": str(request.url),
            "title": content.get("title", "Untitled"),
            "content_length": len(page_text),
            "word_count": len(page_text.split()),
            "images_count": len(content.get("images", [])),
            "tables_count": len(content.get("tables", [])),
            "method_used": result.get("method"),
            "stored_in_knowledge": request.store_in_knowledge,
            "chunks_created": len(chunk_text(page_text)) if len(page_text) > 2000 else 1,
            "timestamp": result.get("timestamp"),
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Widget scrape error: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping error: {str(e)}")


@router.post("/widget/bulk-scrape")
async def widget_bulk_scrape(request: BulkScrapeRequest, background_tasks: BackgroundTasks):
    """Enhanced bulk scraping with better control and filtering"""
    try:
        logger.info(f"Starting bulk scrape from: {request.base_url}")
        discovered_urls = await call_maybe_async(scraper_service.discover_urls, str(request.base_url), request.max_depth, request.max_urls)
        if not discovered_urls:
            return {"status": "no_urls_found", "message": "No URLs discovered from the base URL", "base_url": str(request.base_url)}

        if request.domain_filter:
            filtered_urls = [
                url for url in discovered_urls if request.domain_filter.lower() in urllib.parse.urlparse(url).netloc.lower()
            ]
            discovered_urls = filtered_urls

        background_tasks.add_task(enhanced_bulk_scrape_task, discovered_urls, request.auto_store, request.max_depth)

        return {
            "status": "started",
            "base_url": str(request.base_url),
            "discovered_urls_count": len(discovered_urls),
            "urls_preview": discovered_urls[:5],
            "auto_store": request.auto_store,
            "domain_filter": request.domain_filter,
            "estimated_time_minutes": len(discovered_urls) * 0.5,
        }

    except Exception as e:
        logger.exception(f"Widget bulk scrape error: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk scrape error: {str(e)}")


@router.post("/widget/upload-file")
async def widget_upload_file(file: UploadFile = File(...), store_in_knowledge: bool = True, chunk_large_files: bool = True):
    """Enhanced file upload with robust processing, metadata extraction, and Milvus storage"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        filename = os.path.basename(file.filename)
        guessed = mimetypes.guess_type(filename)
        content_type = file.content_type or (guessed[0] if guessed else None)
        content_type = content_type or "application/octet-stream"

        logger.info(f"Processing uploaded file: {filename} ({content_type})")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        text: Optional[str] = None
        format_type: Optional[str] = None
        metadata: Dict[str, Any] = {}

        # Detect type and extract text
        try:
            if content_type.startswith("text") or filename.lower().endswith((".txt", ".md")):
                text = content.decode("utf-8", errors="replace")
                format_type = "text"

            elif "pdf" in content_type or filename.lower().endswith(".pdf"):
                pdf_reader = PdfReader(BytesIO(content))
                pages_text = []
                try:
                    if getattr(pdf_reader, "is_encrypted", False):
                        try:
                            pdf_reader.decrypt("")
                        except Exception:
                            logger.debug("PDF is encrypted or decryption failed.")
                    for page in getattr(pdf_reader, "pages", []):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            pages_text.append(page_text)
                except Exception as ex_pg:
                    logger.debug(f"PDF page extraction partial failure: {ex_pg}")
                    pages_text = []
                text = "\n\n--- Page Break ---\n\n".join(pages_text)
                format_type = "pdf"
                metadata["total_pages"] = len(getattr(pdf_reader, "pages", []))
                metadata["pages_with_text"] = len(pages_text)

            elif "csv" in content_type or filename.lower().endswith(".csv"):
                csv_text = content.decode("utf-8", errors="replace")
                reader = csv.reader(StringIO(csv_text))
                rows = list(reader)
                if rows:
                    headers = rows[0]
                    data_rows = rows[1:] if len(rows) > 1 else []
                    text_parts = [f"CSV Headers: {', '.join(headers)}", f"Total Rows: {len(data_rows)}", "Sample Data:"]
                    for i, row in enumerate(data_rows[:10]):
                        text_parts.append(f"Row {i+1}: {', '.join(str(cell) for cell in row)}")
                    text = "\n".join(text_parts)
                else:
                    text = csv_text
                    headers = []
                format_type = "csv"
                metadata["total_rows"] = len(rows)
                metadata["columns"] = len(headers) if rows else 0

            elif ("wordprocessingml" in content_type) or filename.lower().endswith(".docx"):
                doc = Document(BytesIO(content))
                paragraphs = [p.text for p in getattr(doc, "paragraphs", []) if p.text and p.text.strip()]
                text = "\n\n".join(paragraphs)
                format_type = "docx"
                metadata["total_paragraphs"] = len(getattr(doc, "paragraphs", []))
                metadata["paragraphs_with_text"] = len(paragraphs)

            elif ("spreadsheetml" in content_type) or filename.lower().endswith(".xlsx"):
                wb = openpyxl.load_workbook(BytesIO(content), data_only=True)
                sheets_text = []
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    sheet_data = []
                    for row in sheet.iter_rows(values_only=True):
                        row_data = [str(cell) if cell is not None else "" for cell in row]
                        if any(cell.strip() for cell in row_data):
                            sheet_data.append(", ".join(row_data))
                    if sheet_data:
                        sheets_text.append(f"Sheet: {sheet_name}\n" + "\n".join(sheet_data))
                text = "\n\n--- Sheet Break ---\n\n".join(sheets_text)
                format_type = "xlsx"
                metadata["total_sheets"] = len(getattr(wb, "sheetnames", []))
                metadata["sheets_with_data"] = len(sheets_text)

            elif content_type in ("text/html", "application/xhtml+xml") or filename.lower().endswith((".html", ".htm")):
                soup = BeautifulSoup(content, "html.parser")
                for element in soup(["script", "style"]):
                    element.decompose()
                text = soup.get_text(separator="\n")
                format_type = "html"

            else:
                try:
                    text = content.decode("utf-8", errors="replace")
                    format_type = "unknown"
                except Exception:
                    raise HTTPException(status_code=415, detail=f"Unsupported file type: {content_type}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="File content is too short or unreadable")

        # normalize whitespace
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        try:
            doc_summary = await call_maybe_async(ai_service.generate_summary, text, max_sentences=4, max_chars=800)
        except Exception as e:
            logger.debug(f"Summary generation failed: {e}")
            doc_summary = (text[:800] + "...") if len(text) > 800 else text

        response: Dict[str, Any] = {
            "filename": filename,
            "format": format_type,
            "content_length": len(text),
            "word_count": len(text.split()),
            "file_size_bytes": len(content),
            "stored_in_knowledge": False,
            "documents_stored": 0,
            "chunks_created": 0,
            "summary": doc_summary,
            "metadata": metadata,
        }

        async def _store_documents_safe(docs: List[Dict[str, Any]]):
            """Safe storage helper for Milvus service"""
            candidates = ["add_documents", "store_documents", "add_docs", "store", "add"]
            func = None
            for name in candidates:
                if hasattr(milvus_service, name):
                    func = getattr(milvus_service, name)
                    break
            if not func:
                raise RuntimeError("milvus_service has no storage method (expected add_documents/store_documents)")

            res = func(docs)
            if inspect.isawaitable(res):
                return await res
            return res

        if store_in_knowledge:
            try:
                if chunk_large_files and len(text) > 2000:
                    chunks = chunk_text(text, chunk_size=1500, overlap=200)
                    documents_to_store = []
                    for i, chunk in enumerate(chunks):
                        documents_to_store.append(
                            {
                                "content": chunk,
                                "url": f"file://{filename}#chunk-{i}",
                                "title": f"{filename} (Part {i+1})",
                                "format": format_type,
                                "timestamp": datetime.now().isoformat(),
                                "source": "widget_upload",
                                "images": [],
                                "metadata": metadata,
                            }
                        )
                    response["chunks_created"] = len(chunks)
                else:
                    documents_to_store = [
                        {
                            "content": text,
                            "url": f"file://{filename}",
                            "title": filename,
                            "format": format_type,
                            "timestamp": datetime.now().isoformat(),
                            "source": "widget_upload",
                            "images": [],
                            "metadata": metadata,
                        }
                    ]
                    response["chunks_created"] = 1

                stored_ids = await _store_documents_safe(documents_to_store)

                if isinstance(stored_ids, (list, tuple, set)):
                    response["documents_stored"] = len(stored_ids)
                    response["stored_in_knowledge"] = len(stored_ids) > 0
                elif isinstance(stored_ids, int):
                    response["documents_stored"] = stored_ids
                    response["stored_in_knowledge"] = stored_ids > 0
                elif stored_ids is None:
                    response["documents_stored"] = 0
                    response["stored_in_knowledge"] = False
                else:
                    try:
                        response["documents_stored"] = len(stored_ids)
                        response["stored_in_knowledge"] = len(stored_ids) > 0
                    except Exception:
                        response["documents_stored"] = 0
                        response["stored_in_knowledge"] = False

            except Exception as e:
                logger.error(f"Error storing file in knowledge base: {e}")
                response["storage_error"] = str(e)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Widget upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.get("/widget/knowledge-stats")
async def widget_knowledge_stats():
    """Enhanced knowledge base statistics for Milvus"""
    try:
        stats = await call_maybe_async(getattr(milvus_service, "get_collection_stats", milvus_service))
        health = await call_maybe_async(getattr(ai_service, "get_service_health", ai_service))

        return {
            "document_count": stats.get("document_count", 0) if isinstance(stats, dict) else 0,
            "collection_status": stats.get("status", "unknown") if isinstance(stats, dict) else "unknown",
            "collection_name": stats.get("collection_name", "unknown") if isinstance(stats, dict) else "unknown",
            "search_config": stats.get("search_config", {}) if isinstance(stats, dict) else {},
            "database": stats.get("database", "milvus") if isinstance(stats, dict) else "milvus",
            "connection": stats.get("connection", {}) if isinstance(stats, dict) else {},
            "indexes": stats.get("indexes", []) if isinstance(stats, dict) else [],
            "embedding_dimension": stats.get("embedding_dimension", 0) if isinstance(stats, dict) else 0,
            "ai_services": health.get("service", {}) if isinstance(health, dict) else {},
            "overall_health": health.get("overall_status", "unknown") if isinstance(health, dict) else "unknown",
            "last_updated": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.exception(f"Widget stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@router.delete("/widget/clear-knowledge")
async def widget_clear_knowledge():
    """Clear Milvus knowledge base with confirmation"""
    try:
        # Using call_maybe_async to support sync/async methods
        await call_maybe_async(getattr(milvus_service, "delete_collection", milvus_service))
        await call_maybe_async(getattr(milvus_service, "initialize", milvus_service))
        return {
            "status": "success",
            "message": "Milvus knowledge base cleared and reinitialized successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception(f"Widget clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


# Background tasks
async def store_document_task(docs: List[Dict[str, Any]]):
    """Background storage task for Milvus service"""
    try:
        func = getattr(milvus_service, "store_documents", None) or getattr(milvus_service, "add_documents", None)
        if not func:
            raise RuntimeError("milvus_service lacks a store_documents/add_documents function")
        res = func(docs)
        if inspect.isawaitable(res):
            await res
        logger.info(f"✅ Stored {len(docs)} docs in Milvus knowledge base")
    except Exception as e:
        logger.error(f"❌ Failed storing docs in Milvus: {e}")


async def enhanced_bulk_scrape_task(urls: List[str], auto_store: bool, max_depth: int):
    """Enhanced bulk scraping with Milvus storage"""
    scraped_count = 0
    stored_count = 0
    error_count = 0
    batch_size = min(5, max(1, len(urls) // 10)) if urls else 1

    logger.info(f"🚀 Starting bulk scrape of {len(urls)} URLs (batch size: {batch_size})")

    for i in range(0, len(urls), batch_size):
        batch = urls[i: i + batch_size]
        batch_start_time = datetime.now()

        scrape_params = {
            "extract_text": True,
            "extract_links": False,
            "extract_images": True,
            "extract_tables": True,
            "scroll_page": True,
            "output_format": "json",
        }

        tasks = [scraper_service.scrape_url(url, scrape_params) for url in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents_to_store = []
        batch_errors = []

        for url, result in zip(batch, results):
            if isinstance(result, Exception):
                error_count += 1
                batch_errors.append(f"{url}: {str(result)}")
                logger.warning(f"⚠️ Error scraping {url}: {str(result)}")
                continue

            if result.get("status") == "success" and result.get("content"):
                scraped_count += 1
                content = result["content"]
                page_text = content.get("text", "")

                if page_text and len(page_text.strip()) >= 100:
                    if auto_store:
                        if len(page_text) > 2500:
                            chunks = chunk_text(page_text, chunk_size=1500, overlap=200)
                            for j, chunk in enumerate(chunks):
                                documents_to_store.append(
                                    {
                                        "content": chunk,
                                        "url": f"{url}#chunk-{j}",
                                        "title": content.get("title", "") or f"Content from {url}",
                                        "format": "text/html",
                                        "timestamp": datetime.now().isoformat(),
                                        "source": "widget_bulk_scrape",
                                        "images": content.get("images", []) if j == 0 else [],
                                    }
                                )
                        else:
                            documents_to_store.append(
                                {
                                    "content": page_text,
                                    "url": url,
                                    "title": content.get("title", "") or f"Content from {url}",
                                    "format": "text/html",
                                    "timestamp": datetime.now().isoformat(),
                                    "source": "widget_bulk_scrape",
                                    "images": content.get("images", []) or [],
                                }
                            )
                else:
                    logger.warning(f"⚠️ Skipping {url}: content too short ({len(page_text)} chars)")
            else:
                error_count += 1
                logger.warning(f"⚠️ Failed to scrape {url}: {result.get('error', 'Unknown error')}")

        if documents_to_store:
            try:
                # Store in Milvus
                store_fn = getattr(milvus_service, "add_documents", None) or getattr(milvus_service, "store_documents", None)
                if not store_fn:
                    raise RuntimeError("No storage function found on milvus_service for bulk store")
                res = store_fn(documents_to_store)
                if inspect.isawaitable(res):
                    stored_ids = await res
                else:
                    stored_ids = res
                stored_count += len(stored_ids) if stored_ids else 0
                logger.info(f"✅ Batch {i//batch_size + 1}: Stored {len(stored_ids) if stored_ids else 0} documents in Milvus")
            except Exception as e:
                logger.exception(f"❌ Error storing batch documents in Milvus: {e}")

        batch_duration = (datetime.now() - batch_start_time).total_seconds()
        logger.info(f"⏱️ Batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1} completed in {batch_duration:.1f}s")

        if batch_errors:
            logger.warning(f"⚠️ Batch errors: {', '.join(batch_errors[:3])}")

        await asyncio.sleep(min(2.0, batch_size * 0.5))

    success_rate = (scraped_count / len(urls)) * 100 if urls else 0
    logger.info(
        f"✅ Bulk scrape completed: {scraped_count}/{len(urls)} scraped ({success_rate:.1f}% success), "
        f"{stored_count} documents stored in Milvus, {error_count} errors"
    )
