import asyncio
import hashlib
import json
import logging
import os
import random
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
import undetected_chromedriver as uc
from bs4 import BeautifulSoup, Tag
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraperService:
    """
    Production-oriented scraper with multiple strategies (trafilatura, requests, selenium),
    robust URL discovery, and structured extraction (text, links, images, tables).

    PRODUCTION FIXES:
    - Graceful handling of permission errors with temp file fallback
    - Proper directory creation with error handling
    - Better error messages and logging
    - Compatible with both ChromaDB and Milvus vector databases
    """

    def __init__(self, request_timeout: int = 30):
        self.session = requests.Session()
        self.request_timeout = request_timeout
        self.ua = self._safe_user_agent()
        self._setup_session()
        self._setup_output_directory()

    def _setup_output_directory(self) -> None:
        """
        PRODUCTION FIX: Setup output directory with proper error handling and fallback.
        Priority: 1. /app/outputs (Docker) 2. ./outputs (Local) 3. /tmp/scraper_outputs (Fallback)
        """
        # Try multiple locations in order of preference
        candidate_dirs = [
            os.getenv("SCRAPER_OUTPUT_DIR", "/app/outputs"),  # Docker environment
            os.path.join(os.getcwd(), "outputs"),             # Local development
            os.path.join(tempfile.gettempdir(), "scraper_outputs")  # System temp fallback
        ]
        
        self.output_dir = None
        self.is_temp_fallback = False
        
        for candidate in candidate_dirs:
            try:
                # Try to create directory
                Path(candidate).mkdir(parents=True, exist_ok=True)
                
                # Test write permissions with a temp file
                test_file = Path(candidate) / f".test_write_{os.getpid()}"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    
                    # Success! Use this directory
                    self.output_dir = Path(candidate)
                    
                    # Set permissions if possible (may fail in containers, that's OK)
                    try:
                        os.chmod(self.output_dir, 0o755)
                    except Exception:
                        pass
                    
                    if candidate == candidate_dirs[-1]:
                        self.is_temp_fallback = True
                        logger.warning(f"⚠️  Using fallback temp directory: {self.output_dir}")
                    else:
                        logger.info(f"✅ Outputs directory: {self.output_dir}")
                    
                    return
                    
                except (PermissionError, OSError):
                    # Can't write to this directory, try next one
                    continue
                    
            except Exception as e:
                logger.debug(f"Could not use directory {candidate}: {e}")
                continue
        
        # If we get here, none worked - use system temp as absolute last resort
        self.output_dir = Path(tempfile.gettempdir())
        self.is_temp_fallback = True
        logger.error(f"❌ All output directories failed, using system temp: {self.output_dir}")

    def _safe_user_agent(self) -> str:
        """Use fake-useragent if available, otherwise a sane fallback UA."""
        try:
            return UserAgent().random
        except Exception:
            return (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )

    def _setup_session(self) -> None:
        self.session.headers.update({
            "User-Agent": self.ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        self.session.request_timeout = self.request_timeout

    async def scrape_url(self, url: str, scrape_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt multiple scraping methods until one succeeds with non-empty text.
        Saves extracted content to output directory in requested format (default JSON).
        Returns a dict with content plus a RAG-ready 'rag_documents' list for direct vector DB ingestion.
        
        Compatible with both ChromaDB and Milvus backends.
        """
        try:
            logger.info(f"🌐 Scraping: {url}")
            self.session.headers["User-Agent"] = self._safe_user_agent()
            await asyncio.sleep(random.uniform(0.2, 0.8))

            for method in ("trafilatura", "requests", "selenium"):
                extractor = getattr(self, f"_scrape_with_{method}", None)
                if extractor is None:
                    continue

                try:
                    content = await extractor(url, scrape_params)
                    text_ok = bool(content and isinstance(content, dict) and (content.get("text") or "").strip())
                    if text_ok:
                        content.setdefault("title", "")
                        content.setdefault("images", [])
                        content.setdefault("links", [])
                        content.setdefault("tables", [])

                        # Chat payload for frontend
                        chat_payload = {
                            "type": "bot",
                            "text": content.get("text", ""),
                            "images": content.get("images", []),
                        }
                        content["chat_payload"] = chat_payload

                        # RAG document format (compatible with Milvus/ChromaDB)
                        rag_doc = {
                            "content": content.get("text", ""),
                            "url": url,
                            "title": content.get("title") or "",
                            "format": "text/html",
                            "source": "web_scraping",
                            "timestamp": datetime.now(),
                            "images": content.get("images", []),
                        }
                        content["rag_documents"] = [rag_doc]

                        # PRODUCTION FIX: Save with better error handling
                        output_format = str(scrape_params.get("output_format", "json")).lower()
                        saved_path = await self._save_to_output_file(url, content, output_format)
                        if saved_path:
                            content["saved_to"] = str(saved_path)

                        logger.info(f"✅ Successfully scraped {url} using {method}")
                        return {
                            "url": url,
                            "method": method,
                            "content": content,
                            "status": "success",
                            "timestamp": datetime.now().isoformat(),
                        }
                except Exception as e:
                    logger.warning(f"⚠️ {method} failed for {url}: {e}", exc_info=False)
                    continue

            logger.error(f"❌ All scraping methods failed for {url}")
            return {
                "url": url,
                "content": None,
                "status": "failed",
                "error": "All methods failed",
            }
        except Exception as e:
            logger.exception(f"❌ Unexpected error scraping {url}: {e}")
            return {"url": url, "content": None, "status": "error", "error": str(e)}

    async def discover_urls(self, base_url: str, max_depth: int = 2, max_urls: int = 100) -> List[str]:
        """
        BFS discovery within the same domain, skipping non-HTML/static files.
        """
        logger.info(f"🔍 Discovering URLs from: {base_url}")
        discovered: set = set()
        to_visit: List[Tuple[str, int]] = [(base_url, 0)]
        visited: set = set()
        base_domain = urlparse(base_url).netloc

        while to_visit and len(discovered) < max_urls:
            current_url, depth = to_visit.pop(0)
            if current_url in visited or depth > max_depth:
                continue
            visited.add(current_url)

            try:
                self.session.headers["User-Agent"] = self._safe_user_agent()
                resp = self.session.get(current_url, timeout=self.request_timeout)
                resp.raise_for_status()

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" not in ctype:
                    continue

                soup = BeautifulSoup(resp.content, "html.parser")
                for a in soup.find_all("a", href=True):
                    href_raw = a["href"].strip()
                    href = urljoin(current_url, href_raw)
                    parsed = urlparse(href)

                    if parsed.netloc != base_domain:
                        continue

                    lower = href.lower()
                    if lower.endswith((
                        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
                        ".svg", ".css", ".js", ".ico", ".zip", ".gz",
                    )):
                        continue

                    if href not in discovered and href not in visited:
                        discovered.add(href)
                        if depth < max_depth:
                            to_visit.append((href, depth + 1))

                await asyncio.sleep(random.uniform(0.1, 0.5))
            except Exception as e:
                logger.warning(f"⚠️ URL discovery failed at {current_url}: {e}", exc_info=False)

        logger.info(f"✅ Discovered {len(discovered)} URLs from {base_url}")
        return list(discovered)

    async def _scrape_with_requests(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape using requests library (fastest method)"""
        resp = self.session.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        self._clean_html(soup)
        return self._extract_common_content(url, soup, params)

    async def _scrape_with_selenium(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape using Selenium (for JavaScript-heavy sites)"""
        options = uc.ChromeOptions()
        for arg in (
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
        ):
            options.add_argument(arg)
        options.add_argument(f"--user-agent={self._safe_user_agent()}")

        driver = None
        try:
            driver = uc.Chrome(options=options)
            driver.set_page_load_timeout(self.request_timeout)
            driver.get(url)
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            if params.get("wait_for_element"):
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, params["wait_for_element"]))
                    )
                except Exception:
                    pass

            if params.get("scroll_page"):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(1.0)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            self._clean_html(soup)
            return self._extract_common_content(url, soup, params)
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    async def _scrape_with_trafilatura(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape using Trafilatura (best for article content)"""
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {}

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True) or ""
        metadata = trafilatura.extract_metadata(downloaded)

        content: Dict[str, Any] = {
            "title": getattr(metadata, "title", "") if metadata else "",
            "author": getattr(metadata, "author", "") if metadata else "",
            "date": getattr(metadata, "date", "") if metadata else "",
            "text": text,
        }

        if params.get("extract_images", False):
            soup = BeautifulSoup(downloaded, "html.parser")
            content["images"] = self._extract_images(url, soup)

        if params.get("extract_links", False):
            soup = BeautifulSoup(downloaded, "html.parser")
            content["links"] = self._extract_links(url, soup)

        if params.get("extract_tables", False):
            soup = BeautifulSoup(downloaded, "html.parser")
            content["tables"] = self._extract_tables(soup)

        return content

    async def _save_to_output_file(self, url: str, content: Dict[str, Any], output_format: str) -> Optional[Path]:
        """
        PRODUCTION FIX: Serialize extracted 'content' to output directory with proper error handling.
        
        Features:
        - Multiple fallback locations
        - Graceful handling of permission errors
        - Temp file fallback as last resort
        - Clear error messages
        
        Returns: Path to saved file, or None if save failed
        """
        try:
            # Generate safe filename
            safe_name = hashlib.md5(url.encode("utf-8")).hexdigest()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "json" if output_format not in ("json", "txt") else output_format
            filename = f"{safe_name}_{ts}.{ext}"
            
            # Try primary output directory first
            if self.output_dir:
                filepath = self.output_dir / filename
                
                try:
                    # Ensure parent directory exists
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Convert content to bytes
                    file_bytes = self._process_to_format_bytes(content, ext)
                    
                    # Try to write file
                    filepath.write_bytes(file_bytes)
                    
                    logger.info(f"✅ Saved to: {filepath}")
                    return filepath
                    
                except PermissionError:
                    # PRODUCTION FIX: Permission denied - try temp fallback
                    logger.warning(f"⚠️  Permission denied for {filepath}, using temp file")
                    
                except OSError as e:
                    # Other OS errors (disk full, etc.)
                    logger.warning(f"⚠️  OS error saving to {filepath}: {e}")
            
            # FALLBACK: Use system temp directory
            temp_path = Path(tempfile.gettempdir()) / filename
            file_bytes = self._process_to_format_bytes(content, ext)
            temp_path.write_bytes(file_bytes)
            
            logger.info(f"✅ Saved to temp: {temp_path}")
            return temp_path
            
        except Exception as e:
            # PRODUCTION FIX: Don't raise - log error and continue
            logger.error(f"❌ Failed saving file: {e}")
            logger.debug(f"Failed to save scraped content for {url}", exc_info=True)
            return None

    def _process_to_format_bytes(self, content: Dict[str, Any], fmt: str) -> bytes:
        """Convert extracted 'content' to bytes for persistence."""
        if fmt == "txt":
            title = content.get("title") or ""
            text = content.get("text") or ""
            payload = (f"{title}\n\n{text}").strip()
            return payload.encode("utf-8", errors="ignore")

        # Default to JSON
        return json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")

    def _clean_html(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements"""
        for tag in ("script", "style", "nav", "footer", "header", "aside"):
            for match in soup.find_all(tag):
                match.decompose()

    def _extract_common_content(self, url: str, soup: BeautifulSoup, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content common to all scraping methods"""
        content: Dict[str, Any] = {"title": soup.title.get_text(strip=True) if soup.title else ""}

        if params.get("extract_text", True):
            main = soup.find("main") or soup.find("article") or soup.find("div", class_=["content", "main", "post"])
            text = (main.get_text(separator=" ", strip=True) if main else soup.get_text(separator=" ", strip=True)) or ""
            content["text"] = text

        if params.get("extract_links", False):
            content["links"] = self._extract_links(url, soup)

        if params.get("extract_images", False):
            content["images"] = self._extract_images(url, soup)

        if params.get("extract_tables", False):
            content["tables"] = self._extract_tables(soup)

        return content

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract all links from the page"""
        links: List[Dict[str, str]] = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if not text:
                continue
            links.append({"url": urljoin(base_url, a["href"]), "text": text})
        return links

    def _extract_images(self, base_url: str, soup: BeautifulSoup) -> List[Dict[str, Any]]:

        images: List[Dict[str, Any]] = []
        seen_urls: set = set()
    
        logger.info(f"🖼️  Starting image extraction from: {base_url}")

        def _add_image(rec: Dict[str, Any]):
        
            if not rec or not rec.get("url"):
                return
        
            url = rec["url"]
        
        # Skip duplicates
            if url in seen_urls:
                return
        
        # Validate image URL
            if not self._is_valid_image_url(url):
                logger.debug(f"Invalid image URL filtered: {url[:80]}")
                return
        
        # ✅ CRITICAL: Ensure absolute URL
            try:
                from urllib.parse import urljoin, urlparse
                absolute_url = urljoin(base_url, url)
                parsed = urlparse(absolute_url)
            
            # Must have scheme and netloc
                if not parsed.scheme or not parsed.netloc:
                    logger.debug(f"Invalid URL structure: {url}")
                    return
            
            # Update record with absolute URL
                rec["url"] = absolute_url
                rec["source_url"] = base_url
            
                seen_urls.add(absolute_url)
                images.append(rec)
            
                logger.debug(f"✅ Added image: {absolute_url[:80]}")
            
            except Exception as e:
                logger.debug(f"Failed to process URL {url}: {e}")
                return

    # =========================================================================
    # 1. Extract from <img> tags (most common)
    # =========================================================================
        for img in soup.find_all("img"):
            rec = self._process_image_tag_enhanced(base_url, img)
            if rec:
            # Enrich with context
                rec = self._enrich_image_with_context(rec, img, soup, base_url)
                _add_image(rec)

    # =========================================================================
    # 2. Extract from <figure> tags (often important images with captions)
    # =========================================================================
        for figure in soup.find_all("figure"):
            img = figure.find("img")
            if not img:
                continue
        
            rec = self._process_image_tag_enhanced(base_url, img)
            if rec:
            # Extract caption from figcaption
                caption_tag = figure.find("figcaption")
                if caption_tag:
                    caption_text = caption_tag.get_text(strip=True)
                    rec["caption"] = caption_text
                # Prepend caption to context text
                    existing_text = rec.get("text", "")
                    rec["text"] = f"{caption_text} | {existing_text}" if existing_text else caption_text
            
                rec = self._enrich_image_with_context(rec, img, soup, base_url)
                _add_image(rec)

    # =========================================================================
    # 3. Extract from <picture> elements (responsive images)
    # =========================================================================
        for picture in soup.find_all("picture"):
        # Try to get the largest/best quality source
            sources = picture.find_all("source")
            best_url = None
        
            for source in sources:
                srcset = source.get("srcset", "")
                if srcset:
                # Parse srcset and get last (usually largest)
                    candidates = []
                    for item in srcset.split(","):
                        parts = item.strip().split()
                        if parts:
                            candidates.append(parts[0])
                    if candidates:
                        best_url = candidates[-1]
                        break
        
        # Fallback to img tag inside picture
            if not best_url:
                img = picture.find("img")
                if img:
                    best_url = img.get("src") or img.get("data-src")
        
            if best_url:
                rec = {
                "url": urljoin(base_url, best_url),
                "alt": picture.find("img").get("alt", "") if picture.find("img") else "",
                "type": "responsive",
                "class": ""
                }
                rec = self._enrich_image_with_context(rec, picture, soup, base_url)
                _add_image(rec)

    # =========================================================================
    # 4. Extract CSS background images (sometimes important visuals)
    # =========================================================================
        for element in soup.find_all(attrs={"style": True}):
            style = element.get("style", "")
            if "background-image" in style:
                matches = re.findall(
                r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', 
                style, 
                flags=re.IGNORECASE
                )
                for match in matches:
                    full_url = urljoin(base_url, match)
                    rec = {
                    "url": full_url,
                    "alt": "background image",
                    "type": "background",
                    "class": ""
                    }
                    rec = self._enrich_image_with_context(rec, element, soup, base_url)
                    _add_image(rec)

    # =========================================================================
    # 5. Quality filtering and ranking
    # =========================================================================
        filtered = self._filter_and_rank_images(images)
    
        logger.info(
        f"✅ Extracted {len(filtered)} quality images from {base_url} "
        f"(filtered from {len(images)} total)"
    )
    
    # Log sample for debugging
        if filtered:
            logger.debug(f"Sample image: {filtered[0].get('url', 'N/A')[:80]}")
    
        return filtered[:40]   # Limit to 40 images per page
    
    

    def _process_image_tag_enhanced(self, base_url: str, img: Tag) -> Optional[Dict[str, Any]]:

        img_url = None
    
    # Priority order for finding image URL
        url_attributes = [
        "src",              # Standard
        "data-src",         # Lazy loading
        "data-lazy-src",    # Lazy loading variant
        "data-original",    # Lazy loading variant
        "data-srcset",      # Responsive variant
        "srcset"            # Responsive standard
        ]
    
    # Try each attribute
        for attr in url_attributes:
            value = img.get(attr)
            if value:
            # Handle srcset (comma-separated list)
                if "srcset" in attr.lower():
                    candidates = []
                    for item in str(value).split(","):
                        parts = item.strip().split()
                        if parts:
                            candidates.append(parts[0])
                    if candidates:
                        img_url = candidates[-1]  # Get largest
                        break
                else:
                    img_url = value
                    break
    
        if not img_url:
            return None

    # Extract metadata
        alt_text = (img.get("alt") or "").strip()
        title = (img.get("title") or "").strip()
        img_class = " ".join(img.get("class", [])) if img.get("class") else ""
    
    # Get dimensions
        width = img.get("width")
        height = img.get("height")

    # Classify image type
        img_type = self._classify_image_type(img_class, alt_text, img_url)

        return {
        "url": img_url,  # Will be converted to absolute in _add_image
        "alt": alt_text or title or "",
        "type": img_type,
        "class": img_class,
        "width": width,
        "height": height,
        }

    
    def _classify_image_type(self, img_class: str, alt_text: str, url: str) -> str:
        lc = img_class.lower()
        la = alt_text.lower() if alt_text else ""
        lu = url.lower()
    
    # Technical/instructional (highest priority for RAG)
        if any(k in la for k in ("diagram", "chart", "graph", "flow", "architecture", "topology")):
            return "diagram"
        if any(k in la for k in ("screenshot", "screen shot", "interface", "ui", "panel", "dashboard")):
            return "screenshot"
        if any(k in lu for k in ("diagram", "chart", "graph", "screenshot", "tutorial")):
            return "diagram"
    
    # Visual content
        if any(k in la for k in ("illustration", "infographic", "drawing", "visual")):
            return "illustration"
        if any(k in la for k in ("photo", "image", "picture")):
            return "photo"
    
    # Branding/decorative (lower priority)
        if any(k in lc for k in ("logo", "icon", "avatar", "brand")):
            return "logo"
        if any(k in lc for k in ("banner", "hero", "header", "promo")):
            return "banner"
    
        return "content"

    

    def _generate_image_description(self, alt: str, img_type: str, rec: Dict[str, Any]) -> str:

    # Use alt text if descriptive enough
        if alt and len(alt) > 15:
            return alt
    
    # Generate based on type
        type_descriptions = {
        "diagram": "Technical diagram or flowchart showing system architecture or process flow",
        "screenshot": "Screenshot of user interface or application screen",
        "illustration": "Illustration or visual graphic explaining a concept",
        "photo": "Photograph or real-world image",
        "logo": "Logo or branding icon",
        "banner": "Banner or promotional header image",
        "content": "Content image providing visual context"
        }
    
        base_desc = type_descriptions.get(img_type, "Visual content image")
    
    # Enhance with URL hints
        url = rec.get("url", "")
        if "login" in url.lower():
            return f"{base_desc} - Login or authentication related"
        if "dashboard" in url.lower():
            return f"{base_desc} - Dashboard or overview screen"
        if "config" in url.lower() or "settings" in url.lower():
            return f"{base_desc} - Configuration or settings interface"
    
        return base_desc
    
    def _filter_and_rank_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        scored_images = []
    
        for img in images:
            score = 0
        
        # Factor 1: Type (0-40 points)
            type_scores = {
            "diagram": 40,
            "screenshot": 35,
            "illustration": 30,
            "photo": 20,
            "content": 15,
            "banner": 5,
            "logo": 0
        }
            score += type_scores.get(img.get("type", "content"), 15)
        
        # Factor 2: Alt text (0-25 points)
            alt = img.get("alt", "")
            if alt:
                alt_len = len(alt)
                if alt_len > 50:
                    score += 25
                elif alt_len > 20:
                    score += 15
                elif alt_len > 5:
                    score += 10
        
        # Factor 3: Context (0-20 points)
            text = img.get("text", "")
            if text:
                text_len = len(text)
                if text_len > 200:
                    score += 20
                elif text_len > 100:
                    score += 15
                elif text_len > 50:
                    score += 10
        
        # Factor 4: Dimensions (0-10 points)
            try:
                width = int(img.get("width", 0))
                height = int(img.get("height", 0))
                if width > 600 and height > 400:
                    score += 10
                elif width > 300 and height > 200:
                    score += 5
            except (ValueError, TypeError):
                pass
        
        # Factor 5: URL quality (0-5 points)
            url = img.get("url", "")
            if any(kw in url.lower() for kw in ["screenshot", "diagram", "tutorial", "guide", "doc"]):
                score += 5
        
            img["quality_score"] = score
            scored_images.append(img)
    
    # Sort by score
        scored_images.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    
    # Log top images for debugging
        if scored_images:
            logger.debug(
                f"Top image: {scored_images[0].get('url', 'N/A')[:60]} "
                f"(score: {scored_images[0].get('quality_score', 0)})"
            )
    
        return scored_images

    def _enrich_image_with_context(self, rec: Dict[str, Any], tag_node: Tag, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """
        Enrich image metadata with surrounding context for better semantic search.
        This is crucial for RAG systems to match images with relevant queries.
        """
        url = rec.get("url", "")
        alt = rec.get("alt", "") or ""
        caption = rec.get("caption", "") or ""

        # Check for figure/figcaption
        figure = tag_node.find_parent("figure") if isinstance(tag_node, Tag) else None
        if figure:
            fc = figure.find("figcaption")
            if fc:
                caption = caption or fc.get_text(strip=True)

        nearby_texts: List[str] = []

        def _gather_nearby(n: Tag, depth: int = 3) -> List[str]:
            """Gather text from nearby elements"""
            texts: List[str] = []
            try:
                # Previous siblings
                for prev in n.previous_siblings:
                    if isinstance(prev, Tag):
                        t = prev.get_text(strip=True)
                        if t:
                            texts.append(t)
                            break
                # Next siblings
                for nxt in n.next_siblings:
                    if isinstance(nxt, Tag):
                        t = nxt.get_text(strip=True)
                        if t:
                            texts.append(t)
                            break
            except Exception:
                pass
            
            # Parent elements
            p = n.find_parent(["p", "li", "td", "div"], limit=depth)
            if p and isinstance(p, Tag):
                tx = p.get_text(separator=" ", strip=True)
                if tx:
                    texts.append(tx)
            
            # Ancestor context
            ancestor = n.parent
            steps = 0
            while ancestor and steps < depth:
                if isinstance(ancestor, Tag):
                    atext = ancestor.get_text(separator=" ", strip=True)
                    if atext:
                        texts.append(atext)
                ancestor = ancestor.parent
                steps += 1
            return texts

        try:
            if isinstance(tag_node, Tag):
                nearby_texts.extend(_gather_nearby(tag_node))
        except Exception:
            pass

        # Assemble context text
        assembled: List[str] = []
        if caption:
            assembled.append(caption)
        if alt:
            assembled.append(alt)
        for t in nearby_texts:
            if t and t not in assembled:
                assembled.append(t)
        
        # Add page snippet as fallback
        try:
            body = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
            if body:
                snippet = body[:400]
                if snippet not in assembled:
                    assembled.append(snippet)
        except Exception:
            pass

        # Join and normalize
        joined = " | ".join(x for x in assembled if x).strip()
        joined = re.sub(r"\s+", " ", joined)

        rec["caption"] = caption
        rec["alt"] = alt
        rec["text"] = joined[:800]  # Limit context length for Milvus VARCHAR field
        rec["url"] = urljoin(base_url, url)
        return rec

    def _is_valid_image_url(self, url: str) -> bool:
 
        if not url or not isinstance(url, str):
            return False
    
        lower = url.lower()
    
    # Filter data URIs
        if lower.startswith("data:"):
            return False
    
    # Filter tracking pixels
        if any(skip in lower for skip in ("1x1", "pixel", "transparent", "spacer", "blank", "empty")):
            return False
    
    # Filter decorative icons
        if (lower.endswith(".svg") or lower.endswith(".gif")):
            if any(skip in lower for skip in ("icon", "logo", "sprite", "emoji")):
                return False
    
    # Must have valid extension
        valid_exts = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
        if not any(ext in lower for ext in valid_exts):
            return False
    
        return True

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from HTML"""
        tables: List[Dict[str, Any]] = []
        for tbl in soup.find_all("table"):
            headers: List[str] = []
            header_row = tbl.find("tr")
            if header_row:
                ths = header_row.find_all("th")
                if ths:
                    headers = [th.get_text(strip=True) for th in ths]
                else:
                    tds = header_row.find_all("td")
                    headers = [td.get_text(strip=True) for td in tds]

            rows_data: List[Any] = []
            for tr in tbl.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                if not cells:
                    continue
                if headers and len(headers) == len(cells):
                    rows_data.append({h: v for h, v in zip(headers, cells)})
                else:
                    rows_data.append(cells)

            if rows_data:
                tables.append({"headers": headers, "rows": rows_data})

        return tables


# Global singleton instance
scraper_service = WebScraperService()