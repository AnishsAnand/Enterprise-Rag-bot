from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from app.services.scraper_service import scraper_service
from app.services.postgres_service import postgres_service

router = APIRouter()

class ScrapeRequest(BaseModel):
    url: HttpUrl
    extract_text: bool = True
    extract_links: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    output_format: str = "json"
    wait_for_element: Optional[str] = None
    scroll_page: bool = False


class BulkScrapeRequest(BaseModel):
    base_url: HttpUrl
    max_depth: int = 2
    max_urls: int = 100
    scrape_params: Dict[str, Any] = {}
    output_format: str = "json"
    store_in_rag: bool = True


@router.post("/scrape")
async def scrape_single_url(request: ScrapeRequest):
    """Scrape a single URL"""
    try:
        scrape_params = {
            'extract_text': request.extract_text,
            'extract_links': request.extract_links,
            'extract_images': request.extract_images,
            'extract_tables': request.extract_tables,
            'wait_for_element': request.wait_for_element,
            'scroll_page': request.scroll_page,
            'output_format': request.output_format
        }

        result = await scraper_service.scrape_url(str(request.url), scrape_params)

        if result.get('status') != 'success':
            raise HTTPException(status_code=400, detail=result.get('error', 'Scraping failed'))

        return {
            'status': 'success',
            'url': str(request.url),
            'content': result.get('content', {}),
            'method_used': result.get('method', 'unknown')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")


@router.post("/bulk-scrape")
async def bulk_scrape(request: BulkScrapeRequest, background_tasks: BackgroundTasks):
    """Discover and scrape multiple URLs from a base URL"""
    try:
        discovered_urls = await scraper_service.discover_urls(
            str(request.base_url),
            request.max_depth
        )
        urls_to_scrape = discovered_urls[:request.max_urls]

        background_tasks.add_task(
            bulk_scrape_task,
            urls_to_scrape,
            request.scrape_params,
            request.output_format,
            request.store_in_rag
        )

        return {
            'status': 'started',
            'discovered_urls_count': len(discovered_urls),
            'urls_to_scrape_count': len(urls_to_scrape),
            'urls_preview': urls_to_scrape[:10]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk scrape failed: {str(e)}")



async def bulk_scrape_task(urls: List[str], scrape_params: Dict[str, Any], output_format: str, store_in_rag: bool):
    """Background task for bulk scraping"""
    scraped_documents = []
    scrape_params['output_format'] = output_format
    batch_size = 5

    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        tasks = [scraper_service.scrape_url(url, scrape_params) for url in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"[Bulk Scrape] Error scraping {url}: {str(result)}")
                continue

            if result.get('status') == 'success':
                content_data = result.get('content', {})
                # FIXED: Extract images from content and include at top level
                images = content_data.get('images', [])
                document = {
                    'url': url,
                    'content': content_data.get('text', ''),
                    'title': content_data.get('title', ''),
                    'format': output_format,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'web_scraping',
                    'images': images,  # CRITICAL: Include images for storage
                    'metadata': content_data
                }
                if images:
                    print(f"[Bulk Scrape] Found {len(images)} images for {url}")
                scraped_documents.append(document)

        await asyncio.sleep(2)

    if store_in_rag and scraped_documents:
        try:
            await postgres_service.add_documents(scraped_documents)
            print(f"[Bulk Scrape] Added {len(scraped_documents)} documents to RAG system")
        except Exception as e:
            print(f"[Bulk Scrape] Error storing documents: {str(e)}")



@router.get("/discover-urls")
async def discover_urls(
    base_url: HttpUrl = Query(..., description="Base URL to discover from"),
    max_depth: int = Query(2, description="Maximum crawl depth"),
    max_urls: int = Query(100, description="Maximum URLs to return")
):
    """Discover URLs from a base URL"""
    try:
        discovered_urls = await scraper_service.discover_urls(str(base_url), max_depth)
        limited_urls = discovered_urls[:max_urls]

        return {
            'base_url': str(base_url),
            'total_discovered': len(discovered_urls),
            'returned_count': len(limited_urls),
            'urls': limited_urls
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL discovery failed: {str(e)}")


@router.get("/scraping-status")
async def get_scraping_status():
    """Get current scraping status and RAG stats"""
    try:
        milvus_stats = await postgres_service.get_collection_stats()
        return {
            'status': 'active',
            'documents_stored': milvus_stats.get('document_count', 0),
            'collection_name': milvus_stats.get('collection_name', 'default')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch scraping status: {str(e)}")
