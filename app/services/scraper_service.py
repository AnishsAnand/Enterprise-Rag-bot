"""
Web Scraper Service - Multi-strategy content extraction for RAG ingestion.
Supports Trafilatura, Requests, and Selenium for maximum coverage.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
from datetime import datetime
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

OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIRECTORY", "/tmp/rag_outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraperService:
    """Multi-strategy scraper with URL discovery and structured extraction."""

    def __init__(self, request_timeout: int = 30):
        self.session = requests.Session()
        self.request_timeout = request_timeout
        self.ua = self._safe_user_agent()
        self._setup_session()

    def _safe_user_agent(self) -> str:
        try:
            return UserAgent().random
        except Exception:
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"

    def _setup_session(self) -> None:
        self.session.headers.update({
            "User-Agent": self.ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })

    async def scrape_url(self, url: str, scrape_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape URL using multiple methods until success.
        Returns content with RAG-ready document format.
        """
        try:
            logger.info(f"🌐 Scraping: {url}")
            self.session.headers["User-Agent"] = self._safe_user_agent()
            await asyncio.sleep(random.uniform(0.2, 0.8))

            for method in ("trafilatura", "requests", "selenium"):
                extractor = getattr(self, f"_scrape_with_{method}", None)
                if not extractor:
                    continue

                try:
                    content = await extractor(url, scrape_params)
                    if content and isinstance(content, dict) and (content.get("text") or "").strip():
                        content.setdefault("title", "")
                        content.setdefault("images", [])
                        content.setdefault("links", [])
                        content.setdefault("tables", [])

                        # Chat payload for frontend
                        content["chat_payload"] = {
                            "type": "bot",
                            "text": content.get("text", ""),
                            "images": content.get("images", []),
                        }

                        # RAG document format
                        content["rag_documents"] = [{
                            "content": content.get("text", ""),
                            "url": url,
                            "title": content.get("title") or "",
                            "format": "text/html",
                            "source": "web_scraping",
                            "timestamp": datetime.now().isoformat(),
                            "images": content.get("images", []),
                        }]

                        output_format = str(scrape_params.get("output_format", "json")).lower()
                        await self._save_to_output_file(url, content, output_format)

                        logger.info(f"✅ Scraped {url} using {method}")
                        return {"url": url, "method": method, "content": content, "status": "success", "timestamp": datetime.now().isoformat()}
                except Exception as e:
                    logger.warning(f"⚠️ {method} failed for {url}: {e}")
                    continue

            logger.error(f"❌ All methods failed for {url}")
            return {"url": url, "content": None, "status": "failed", "error": "All methods failed"}
        except Exception as e:
            logger.exception(f"❌ Scraping error {url}: {e}")
            return {"url": url, "content": None, "status": "error", "error": str(e)}

    async def discover_urls(self, base_url: str, max_depth: int = 2, max_urls: int = 100) -> List[str]:
        """BFS discovery within the same domain."""
        logger.info(f"🔍 Discovering URLs from: {base_url}")
        discovered, visited = set(), set()
        to_visit: List[Tuple[str, int]] = [(base_url, 0)]
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

                if "text/html" not in (resp.headers.get("Content-Type") or "").lower():
                    continue

                soup = BeautifulSoup(resp.content, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urljoin(current_url, a["href"].strip())
                    parsed = urlparse(href)
                    if parsed.netloc != base_domain:
                        continue
                    if href.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js", ".ico", ".zip", ".gz")):
                        continue
                    if href not in discovered and href not in visited:
                        discovered.add(href)
                        if depth < max_depth:
                            to_visit.append((href, depth + 1))

                await asyncio.sleep(random.uniform(0.1, 0.5))
            except Exception as e:
                logger.warning(f"⚠️ Discovery failed at {current_url}: {e}")

        logger.info(f"✅ Discovered {len(discovered)} URLs")
        return list(discovered)

    # ── Scraping Methods ────────────────────────────────────────────────────
    async def _scrape_with_requests(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fast scraping using requests library."""
        resp = self.session.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        self._clean_html(soup)
        return self._extract_content(url, soup, params)

    async def _scrape_with_selenium(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scraping for JavaScript-heavy sites."""
        options = uc.ChromeOptions()
        for arg in ("--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--disable-extensions"):
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
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, params["wait_for_element"])))
                except Exception:
                    pass

            if params.get("scroll_page"):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(1.0)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            self._clean_html(soup)
            return self._extract_content(url, soup, params)
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    async def _scrape_with_trafilatura(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Article-focused extraction using Trafilatura."""
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {}

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True) or ""
        metadata = trafilatura.extract_metadata(downloaded)

        content = {
            "title": getattr(metadata, "title", "") if metadata else "",
            "author": getattr(metadata, "author", "") if metadata else "",
            "date": getattr(metadata, "date", "") if metadata else "",
            "text": text,
        }

        soup = BeautifulSoup(downloaded, "html.parser")
        if params.get("extract_images", False):
            content["images"] = self._extract_images(url, soup)
        if params.get("extract_links", False):
            content["links"] = self._extract_links(url, soup)
        if params.get("extract_tables", False):
            content["tables"] = self._extract_tables(soup)

        return content

    # ── Content Extraction ──────────────────────────────────────────────────
    def _clean_html(self, soup: BeautifulSoup) -> None:
        for tag in ("script", "style", "nav", "footer", "header", "aside"):
            for match in soup.find_all(tag):
                match.decompose()

    def _extract_content(self, url: str, soup: BeautifulSoup, params: Dict[str, Any]) -> Dict[str, Any]:
        content = {"title": soup.title.get_text(strip=True) if soup.title else ""}

        if params.get("extract_text", True):
            main = soup.find("main") or soup.find("article") or soup.find("div", class_=["content", "main", "post"])
            content["text"] = (main.get_text(separator=" ", strip=True) if main else soup.get_text(separator=" ", strip=True)) or ""

        if params.get("extract_links", False):
            content["links"] = self._extract_links(url, soup)
        if params.get("extract_images", False):
            content["images"] = self._extract_images(url, soup)
        if params.get("extract_tables", False):
            content["tables"] = self._extract_tables(soup)

        return content

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> List[Dict[str, str]]:
        links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            if text:
                links.append({"url": urljoin(base_url, a["href"]), "text": text})
        return links

    def _extract_images(self, base_url: str, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract images with context for semantic matching."""
        images, seen = [], set()

        def add_image(rec):
            if rec and rec.get("url") and rec["url"] not in seen and self._is_valid_image_url(rec["url"]):
                seen.add(rec["url"])
                images.append(rec)

        # <img> tags
        for img in soup.find_all("img"):
            rec = self._process_image_tag(base_url, img)
            if rec:
                rec = self._enrich_image_context(rec, img, soup, base_url)
                add_image(rec)

        # <figure> tags
        for figure in soup.find_all("figure"):
            img = figure.find("img")
            if not img:
                continue
            rec = self._process_image_tag(base_url, img)
            if rec:
                if fc := figure.find("figcaption"):
                    rec["caption"] = fc.get_text(strip=True)
                rec = self._enrich_image_context(rec, img, soup, base_url)
                add_image(rec)

        # CSS background images
        for el in soup.find_all(attrs={"style": True}):
            style = el.get("style", "")
            if "background-image" in style:
                for match in re.findall(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style, re.IGNORECASE):
                    rec = {"url": urljoin(base_url, match), "alt": "background image", "type": "background"}
                    rec = self._enrich_image_context(rec, el, soup, base_url)
                    add_image(rec)

        return images[:40]

    def _process_image_tag(self, base_url: str, img: Tag) -> Optional[Dict[str, Any]]:
        img_url = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not img_url and img.get("srcset"):
            candidates = [c.strip().split(" ")[0] for c in img.get("srcset").split(",") if c.strip()]
            if candidates:
                img_url = candidates[-1]
        if not img_url:
            return None

        full_url = urljoin(base_url, img_url)
        alt = (img.get("alt") or img.get("title") or "").strip()
        img_class = " ".join(img.get("class", [])) if img.get("class") else ""

        # Classify type
        img_type = "content"
        lc = img_class.lower()
        if any(k in lc for k in ("logo", "icon", "avatar")):
            img_type = "logo/icon"
        elif any(k in lc for k in ("banner", "hero")):
            img_type = "banner"
        elif any(k in alt.lower() for k in ("diagram", "chart", "graph")):
            img_type = "diagram"

        return {"url": full_url, "alt": alt, "type": img_type, "class": img_class}

    def _enrich_image_context(self, rec: Dict[str, Any], tag: Tag, soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
        """Add surrounding text for semantic search."""
        alt = rec.get("alt", "") or ""
        caption = rec.get("caption", "") or ""

        # Get nearby text
        nearby = []
        if isinstance(tag, Tag):
            for sib in list(tag.previous_siblings)[:2] + list(tag.next_siblings)[:2]:
                if isinstance(sib, Tag) and (t := sib.get_text(strip=True)):
                    nearby.append(t)
                    break
            if p := tag.find_parent(["p", "li", "td", "div"]):
                if t := p.get_text(separator=" ", strip=True):
                    nearby.append(t)

        # Assemble context
        parts = [p for p in [caption, alt] + nearby if p]
        if soup.body:
            parts.append(soup.body.get_text(separator=" ", strip=True)[:400])

        rec["text"] = re.sub(r"\s+", " ", " | ".join(p for p in parts if p))[:800]
        rec["url"] = urljoin(base_url, rec.get("url", ""))
        return rec

    def _is_valid_image_url(self, url: str) -> bool:
        if not url or url.lower().startswith("data:"):
            return False
        lower = url.lower()
        if any(s in lower for s in ("1x1", "pixel", "transparent", "spacer", "blank")):
            return False
        return any(ext in lower for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"))

    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        tables = []
        for tbl in soup.find_all("table"):
            headers = []
            if header_row := tbl.find("tr"):
                ths = header_row.find_all("th")
                headers = [th.get_text(strip=True) for th in ths] if ths else [td.get_text(strip=True) for td in header_row.find_all("td")]

            rows = []
            for tr in tbl.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(dict(zip(headers, cells)) if headers and len(headers) == len(cells) else cells)

            if rows:
                tables.append({"headers": headers, "rows": rows})

        return tables

    # ── File Output ─────────────────────────────────────────────────────────
    async def _save_to_output_file(self, url: str, content: Dict[str, Any], output_format: str) -> None:
        try:
            safe_name = hashlib.md5(url.encode()).hexdigest()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = output_format if output_format in ("json", "txt") else "json"
            filepath = os.path.join(OUTPUT_DIR, f"{safe_name}_{ts}.{ext}")

            data = content.get("text", "") if ext == "txt" else json.dumps(content, ensure_ascii=False, indent=2)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(data)

            logger.debug(f"💾 Saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")


# Global singleton
scraper_service = WebScraperService()
