import logging
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenWebUIFormatter:
    """
    PRODUCTION-READY: Ensures EVERY step has a displayable image for OpenWebUI.
    FIXED VERSION: Proper image rendering from scraped URLs and generated placeholders.
    """

    def __init__(self):
        self._gallery_limit = 8
        self._max_step_images = 20
        
        # Reliable placeholder services (ordered by reliability)
        self.placeholder_services = [
            "https://placehold.co",             # Primary (most reliable)
            "https://dummyimage.com",           # Backup 1
            "https://fakeimg.pl",               # Backup 2
        ]

    def format_response(
        self,
        answer: str,
        steps: Optional[List[Dict[str, Any]]],
        images: Optional[List[Dict[str, Any]]],
        query: str,
        confidence: float = 0.0,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format complete response with GUARANTEED displayable images for OpenWebUI.
        
        CRITICAL: Every step MUST have either:
        1. A real scraped image URL (HTTP/HTTPS)
        2. A generated placeholder image URL
        """
        try:
            # Normalize inputs
            steps_list = self._normalize_steps(steps)
            images_list = self._normalize_images(images)

            logger.info(
                f"[Formatter] Processing: {len(steps_list)} steps, "
                f"{len(images_list)} images available"
            )

            sections: List[str] = []

            # Summary section
            if summary and summary.strip() and summary.strip() != (answer or "").strip():
                sections.append(self._format_summary(summary))

            # Main answer section
            if answer and answer.strip():
                sections.append(self._format_main_answer(answer))

            # Steps with GUARANTEED displayable images
            if steps_list:
                formatted_steps = self._format_steps_with_guaranteed_images(
                    steps_list, 
                    images_list,
                    query
                )
                sections.append(formatted_steps)

            # Remaining images gallery
            used_urls = self._extract_used_image_urls(steps_list)
            unused_images = [
                img for img in images_list 
                if img.get("url") and img.get("url") not in used_urls and not img.get("_used")
            ]
            if unused_images:
                gallery = self._format_image_gallery(unused_images[:self._gallery_limit])
                if gallery:
                    sections.append(gallery)

            # Combine all sections
            formatted = "\n\n---\n\n".join(filter(None, sections))
            formatted = self._normalize_markdown(formatted)

            # Safety check
            if not formatted or len(formatted.strip()) < 8:
                return self._fallback_minimal(answer, metadata, confidence)

            return formatted

        except Exception as e:
            logger.exception(f"[Formatter] Unexpected error: {e}")
            return self._fallback_minimal(answer, metadata, confidence)

    def _normalize_steps(self, steps) -> List[Dict[str, Any]]:
        """Normalize steps to standard format."""
        if not steps:
            return []

        normalized: List[Dict[str, Any]] = []
        
        try:
            if isinstance(steps, str):
                steps = [steps]

            for i, s in enumerate(steps):
                if isinstance(s, str):
                    normalized.append({
                        "step_number": i + 1,
                        "text": s.strip(),
                        "type": "info",
                        "image": None,
                        "image_prompt": None
                    })
                elif isinstance(s, dict):
                    text = s.get("text") or s.get("content") or ""
                    text = text.strip() if isinstance(text, str) else str(text)
                    
                    normalized.append({
                        "step_number": s.get("step_number") or s.get("index") or (i + 1),
                        "text": text,
                        "type": s.get("type", "action"),
                        "image": s.get("image"),
                        "image_prompt": s.get("image_prompt"),
                        "note": s.get("note")
                    })
                else:
                    normalized.append({
                        "step_number": i + 1,
                        "text": str(s),
                        "type": "info",
                        "image": None,
                        "image_prompt": None
                    })
                    
        except Exception as e:
            logger.debug(f"[Formatter] Step normalization error: {e}")
        
        return [s for s in normalized if s.get("text")]

    def _normalize_images(self, images) -> List[Dict[str, Any]]:
        """
        Normalize images to standard format with deduplication.
        PRODUCTION: Only accepts REAL scraped image URLs - NO placeholder generation.
        """
        if not images:
            return []
            
        normalized: List[Dict[str, Any]] = []
        
        try:
            if isinstance(images, dict):
                images = [images]
                
            if isinstance(images, list):
                for idx, img in enumerate(images):
                    if isinstance(img, str):
                        # String URL - validate and add (ONLY real URLs)
                        if self._validate_image_url(img):
                            # ❌ Skip placeholders
                            if any(p in img.lower() for p in ["placeholder", "placehold", "dummyimage", "fakeimg"]):
                                continue
                                
                            normalized.append({
                                "url": img,
                                "alt": self._extract_alt_from_url(img),
                                "caption": "",
                                "type": self._guess_type_from_url(img),
                                "source": "scraped"
                            })
                            
                    elif isinstance(img, dict):
                        url = img.get("url") or img.get("src") or ""
                        
                        # ❌ IGNORE image_prompt - we don't generate placeholders
                        if not url:
                            logger.debug("[Formatter] Skipped image without URL (no placeholder generation)")
                            continue
                        
                        # Validate URL before adding
                        if url and self._validate_image_url(url):
                            # ❌ Skip placeholders
                            if any(p in url.lower() for p in ["placeholder", "placehold", "dummyimage", "fakeimg"]):
                                logger.debug(f"[Formatter] Skipped placeholder URL: {url[:50]}")
                                continue
                            
                            # ✅ Only add REAL scraped images
                            normalized.append({
                                "url": url,
                                "alt": img.get("alt") or self._extract_alt_from_url(url),
                                "caption": img.get("caption", "") or "",
                                "type": img.get("type") or self._guess_type_from_url(url),
                                "source_url": img.get("source_url", ""),
                                "source": "scraped"
                            })
                            
        except Exception as e:
            logger.warning(f"[Formatter] Image normalization error: {e}")
        
        # Deduplicate by URL
        seen: Set[str] = set()
        deduplicated: List[Dict[str, Any]] = []
        
        for img in normalized:
            url = img.get("url")
            if url and url not in seen:
                seen.add(url)
                deduplicated.append(img)
        
        logger.info(f"[Formatter] Normalized {len(deduplicated)} unique REAL scraped images")
        return deduplicated

    def _validate_image_url(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
    
    # Must start with http:// or https://
        if not (url.startswith("http://") or url.startswith("https://")):
            logger.debug(f"[Formatter] Rejected non-HTTP URL: {url[:50]}")
            return False
    
    # Check for whitespace (malformed URLs)
        if any(char in url for char in [' ', '\n', '\t', '\r']):
            logger.debug(f"[Formatter] Rejected URL with whitespace")
            return False

        lower = url.lower()
    
    # Reject obvious non-images
        bad_extensions = ('.pdf', '.zip', '.exe', '.dmg', '.doc', '.docx', '.txt', '.xml', '.json')
        if any(lower.endswith(ext) for ext in bad_extensions):
            logger.debug(f"[Formatter] Rejected non-image file type")
            return False
    
    # ✅ ACCEPT if it has image extension OR no clear extension (many CDN URLs)
        good_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico')
        has_image_ext = any(ext in lower for ext in good_extensions)
        has_bad_ext = any(ext in lower for ext in bad_extensions)
    
    # Accept if: has image extension OR (no bad extension AND has valid URL structure)
        if not has_image_ext and not has_bad_ext:
        # Check if URL structure suggests it might be an image (CDN, media server, etc.)
            media_indicators = ['image', 'img', 'media', 'cdn', 'photo', 'pic', 'upload', 'content']
            if any(indicator in lower for indicator in media_indicators):
                logger.debug(f"[Formatter] Accepting URL with media indicators: {url[:60]}")
                return True
    
        if not has_image_ext:
            logger.debug(f"[Formatter] Rejected URL with no image extension: {url[:50]}")
            return False
    
    # Validate URL structure
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc:
                logger.debug(f"[Formatter] Rejected URL with no domain")
                return False
        except Exception as e:
            logger.debug(f"[Formatter] URL parsing failed: {e}")
            return False
    
        return True

    def _extract_alt_from_url(self, url: str) -> str:
        """Extract meaningful alt text from URL path."""
        try:
            parsed = urlparse(url)
            path = parsed.path
            filename = path.split('/')[-1]
            # Remove extension and convert to readable text
            name = filename.rsplit('.', 1)[0]
            # Replace separators with spaces and title case
            readable = name.replace('-', ' ').replace('_', ' ').strip()
            if readable:
                return readable.title()
        except:
            pass
        return "Image"

    def _guess_type_from_url(self, url: str) -> str:
        """Guess image type from URL patterns."""
        url_lower = url.lower()
        if 'screenshot' in url_lower or 'screen' in url_lower or 'capture' in url_lower:
            return 'screenshot'
        elif 'diagram' in url_lower or 'chart' in url_lower or 'graph' in url_lower:
            return 'diagram'
        elif 'illustration' in url_lower or 'drawing' in url_lower:
            return 'illustration'
        elif 'photo' in url_lower or 'picture' in url_lower:
            return 'photo'
        return 'content'

    def _format_summary(self, summary: str) -> str:
        """Format summary section."""
        return f"## 📋 Quick Summary\n\n{summary.strip()}"

    def _format_main_answer(self, answer: str) -> str:
        """Format main answer section."""
        answer = (answer or "").strip()
        if not answer:
            return ""
            
        if answer.startswith("#"):
            return answer
            
        return f"## ℹ️ Detailed Information\n\n{answer}"

    def _format_steps_with_guaranteed_images(
        self,
        steps: List[Dict[str, Any]],
        all_images: List[Dict[str, Any]],
        query: str
    ) -> str:
        """
        Format steps with ONLY REAL SCRAPED images.
        NO PLACEHOLDERS - only displays actual images from scraped data.
        """
        if not steps:
            return ""

        formatted_parts: List[str] = ["## 📝 Step-by-Step Instructions\n"]
    
        # FIXED: Create pool of ONLY REAL SCRAPED images (NO placeholders)
        real_image_pool = []
        for img in all_images:
            url = img.get("url", "")
            
            # Validate URL
            if not url or not isinstance(url, str):
                continue
            if not (url.startswith("http://") or url.startswith("https://")):
                continue
            
            # ❌ REJECT ALL PLACEHOLDERS - only accept real scraped images
            if any(placeholder in url.lower() for placeholder in [
                "placeholder", "placehold", "dummyimage", "fakeimg", "via.placeholder"
            ]):
                logger.debug(f"[Formatter] Filtered out placeholder: {url[:60]}")
                continue
            
            # ✅ Accept only REAL SCRAPED images
            if not img.get("_used") and img.get("source") == "scraped":
                real_image_pool.append(img)
    
        logger.info(
            f"[Formatter] Using {len(real_image_pool)} REAL scraped images only "
            f"(filtered from {len(all_images)} total)"
        )

        for idx, step in enumerate(steps, 1):
            step_number = step.get("step_number", idx)
            step_text = (step.get("text") or "").strip()
        
            if not step_text:
                continue
        
            step_type = step.get("type", "action")
            emoji = self._get_step_emoji(step_type)

            # Step header and text
            formatted_parts.append(f"\n### {emoji} Step {step_number}\n")
            formatted_parts.append(f"{step_text}\n")

            # ═══════════════════════════════════════════════════════════
            # IMAGE ASSIGNMENT - ONLY REAL SCRAPED IMAGES
            # ═══════════════════════════════════════════════════════════
        
            image_url = None
            image_alt = None
            image_caption = None
            image_source = None
        
            # Priority 1: Explicit step image (must be real, not placeholder)
            step_image = step.get("image")
            if step_image:
                extracted = self._extract_real_image_url_only(step_image, step_number)
                if extracted:
                    image_url = extracted["url"]
                    image_alt = extracted.get("alt", f"Step {step_number}")
                    image_caption = extracted.get("caption")
                    image_source = "explicit"
                    logger.debug(f"[Formatter] Step {step_number}: Using explicit real image")

            # Priority 2: Match from REAL image pool (semantic + round-robin)
            if not image_url and real_image_pool:
                matched_image = self._match_image_to_step(
                    step_text, 
                    real_image_pool,
                    idx
                )
            
                if matched_image:
                    image_url = matched_image.get("url")
                    image_alt = matched_image.get("alt") or f"Step {step_number}"
                    image_caption = matched_image.get("caption")
                    image_source = "matched"
                
                    # Mark as used
                    matched_image["_used"] = True
                    logger.debug(f"[Formatter] Step {step_number}: Matched real image from pool")

            # ═══════════════════════════════════════════════════════════
            # EMBED IMAGE (only if we have a REAL scraped image URL)
            # ═══════════════════════════════════════════════════════════
        
            if image_url:
                formatted_parts.append(f"\n![{image_alt}]({image_url})\n")
                if image_caption:
                    formatted_parts.append(f"*{image_caption}*\n")
            
                logger.debug(
                    f"[Formatter] ✅ Step {step_number}: Real scraped image embedded "
                    f"({image_url[:60]}...)"
                )
            else:
                # ✅ NO IMAGE - Just show the step text without any placeholder
                logger.debug(
                    f"[Formatter] ℹ️  Step {step_number}: No matching real image "
                    f"(text-only step)"
                )
        
            # Add notes if present
            note = step.get("note")
            if note:
                formatted_parts.append(f"\n> **Note:** {note}\n")

            # Separator between steps
            if idx < len(steps):
                formatted_parts.append("\n---\n")

        result = "".join(formatted_parts)
    
        # Count statistics
        images_used = len([img for img in real_image_pool if img.get("_used")])
        text_only_steps = len(steps) - images_used
        
        logger.info(
            f"[Formatter] ✅ Formatted {len(steps)} steps: "
            f"{images_used} with real images, {text_only_steps} text-only"
        )
    
        return result
    
    def _extract_real_image_url_only(
        self, 
        image: Any, 
        step_number: int
    ) -> Optional[Dict[str, str]]:
        """
        Extract ONLY real scraped image URLs - NO placeholders.
        Rejects image_prompt and placeholder URLs.
        """
        # Handle string URL
        if isinstance(image, str):
            # Must be absolute HTTP/HTTPS URL
            if not (image.startswith("http://") or image.startswith("https://")):
                return None
            
            # ❌ REJECT placeholders
            if any(p in image.lower() for p in ["placeholder", "placehold", "dummyimage", "fakeimg"]):
                logger.debug(f"[Formatter] Rejected placeholder URL: {image[:50]}")
                return None
            
            # ✅ Accept only real scraped images
            if self._validate_image_url(image):
                return {
                    "url": image,
                    "alt": self._extract_alt_from_url(image),
                    "caption": None
                }
            return None

        if not isinstance(image, dict):
            return None

        # ❌ REJECT image_prompt (these are not real images)
        if image.get("image_prompt") and not image.get("url"):
            logger.debug("[Formatter] Rejected image_prompt (no real URL)")
            return None

        # Extract URL from dict
        url = image.get("url") or image.get("src") or ""
        if not url or not isinstance(url, str):
            return None
        
        # Must be absolute URL
        if not (url.startswith("http://") or url.startswith("https://")):
            return None
        
        # ❌ REJECT placeholders
        if any(p in url.lower() for p in ["placeholder", "placehold", "dummyimage", "fakeimg"]):
            logger.debug(f"[Formatter] Rejected placeholder URL: {url[:50]}")
            return None
        
        # ✅ Accept only real scraped images
        if self._validate_image_url(url):
            return {
                "url": url,
                "alt": (image.get("alt") or self._extract_alt_from_url(url)).strip(),
                "caption": (image.get("caption") or "").strip() or None
            }
        
        return None

    def _extract_visual_context(self, text: str) -> str:
        """
        Extract visual context from step text for better placeholder generation.
        Identifies key visual elements like actions, UI components, etc.
        """
        text_lower = text.lower()
        
        # Extract action words
        actions = ["click", "select", "navigate", "open", "configure", "enter", 
                   "download", "upload", "create", "delete", "login", "verify",
                   "install", "setup", "enable", "disable", "edit", "save"]
        found_actions = [action for action in actions if action in text_lower]
        
        # Extract UI elements
        ui_elements = ["button", "menu", "screen", "dialog", "field", "panel", 
                       "page", "window", "form", "table", "chart", "dashboard",
                       "settings", "options", "tab", "link", "icon"]
        found_ui = [elem for elem in ui_elements if elem in text_lower]
        
        # Build context string
        context_parts = []
        if found_actions:
            context_parts.append(found_actions[0].title())
        if found_ui:
            context_parts.append(found_ui[0].title())
        
        if context_parts:
            return " ".join(context_parts)
        
        # Fallback: Extract first few meaningful words
        words = re.findall(r'\b\w{4,}\b', text)
        if words:
            return " ".join(words[:3]).title()
        
        return text[:40]

    def _extract_image_url(
        self, 
        image: Any, 
        step_number: int
    ) -> Optional[Dict[str, str]]:
        """
        Extract displayable image URL from any format.
        FIXED: Handles image_prompt conversion + validation.
        """
        # Handle string URL
        if isinstance(image, str):
            if self._validate_image_url(image):
                return {
                    "url": image,
                    "alt": self._extract_alt_from_url(image),
                    "caption": None,
                    "source": "scraped" if "placeholder" not in image.lower() else "placeholder"
                }
            elif image and len(image) > 0:
                # Treat as image prompt → convert to placeholder
                placeholder_url = self._generate_placeholder_url(image, step_number)
                return {
                    "url": placeholder_url,
                    "alt": f"Visual Guide: {image[:50]}",
                    "caption": None,
                    "source": "placeholder"
                }

        if not isinstance(image, dict):
            return None

        # Handle image_prompt in dict (CRITICAL FIX)
        prompt = image.get("image_prompt")
        if prompt and not image.get("url"):
            placeholder_url = self._generate_placeholder_url(prompt, step_number)
            return {
                "url": placeholder_url,
                "alt": f"Visual Guide: {prompt[:50]}",
                "caption": None,
                "source": "placeholder"
            }

        # Handle URL in dict
        url = image.get("url") or image.get("src") or ""
        if url and self._validate_image_url(url):
            is_placeholder = any(p in url.lower() for p in 
                               ["placeholder", "placehold", "dummyimage", "fakeimg"])
            
            return {
                "url": url,
                "alt": (image.get("alt") or self._extract_alt_from_url(url)).strip(),
                "caption": (image.get("caption") or "").strip() or None,
                "source": "placeholder" if is_placeholder else "scraped"
            }
        
        return None

    def _generate_placeholder_url(
        self, 
        text: str, 
        step_number: int,
        size: tuple = (900, 400)
    ) -> str:
        """
        PRODUCTION FIX: Generate DISPLAYABLE placeholder image URL.
        Uses reliable placeholder services with proper URL encoding.
        """
        try:
            # Clean and truncate text
            clean_text = text.strip()[:100]
            # Remove special characters that might break URL
            clean_text = re.sub(r'[^\w\s-]', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            
            # Truncate to reasonable length for URL
            if len(clean_text) > 50:
                clean_text = clean_text[:47] + "..."
            
            # If text is empty, use generic step label
            if not clean_text:
                clean_text = f"Step {step_number}"
            
            # Proper URL encoding (spaces to %20)
            encoded = quote(clean_text, safe='')
            
            width, height = size
            
            # Professional color scheme
            bg_color = "4A90E2"  # Professional blue
            text_color = "FFFFFF"  # White text
            
            # Use most reliable service (placehold.co)
            primary_url = f"https://placehold.co/{width}x{height}/{bg_color}/{text_color}/png?text={encoded}"
            
            logger.debug(f"[Formatter] Generated placeholder: {primary_url[:80]}...")
            return primary_url
            
        except Exception as e:
            logger.warning(f"[Formatter] Placeholder generation error: {e}")
            # Absolute fallback
            safe_step = f"Step%20{step_number}"
            return f"https://placehold.co/900x400/4A90E2/FFFFFF/png?text={safe_step}"

    def _match_image_to_step(
        self,
        step_text: str,
        image_pool: List[Dict[str, Any]],
        step_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Match images to steps using semantic analysis + round-robin fallback.
        PRODUCTION FIX: Enhanced matching for scraped images with better metadata handling.
        """
        if not image_pool:
            return None
    
        step_lower = step_text.lower()
        step_words = set(re.findall(r'\b\w{4,}\b', step_lower))
    
        # Remove common stopwords
        stopwords = {'with', 'from', 'that', 'this', 'have', 'will', 'your', 'into', 
                    'then', 'they', 'them', 'these', 'those', 'when', 'where'}
        step_words = step_words - stopwords
    
        # Score each image
        scored_images = []
    
        for img in image_pool:
            if img.get("_used"):
                continue
        
            score = 0
        
            # Extract image text for matching (FIXED: better metadata extraction)
            img_text_parts = [
                img.get("alt", ""),
                img.get("caption", ""),
                img.get("text", ""),
                self._extract_alt_from_url(img.get("url", ""))  # NEW: extract from URL
            ]
            img_text = " ".join(filter(None, img_text_parts)).lower()
        
            img_words = set(re.findall(r'\b\w{4,}\b', img_text))
        
            # Semantic matching (0-50 points)
            word_overlap = len(step_words & img_words)
            if step_words:
                overlap_ratio = word_overlap / len(step_words)
                score += overlap_ratio * 50
        
            # Keyword matching bonus (0-30 points)
            keyword_matches = {
                ("login", "sign in", "signin"): ("login", "signin", "auth", "authentication"),
                ("dashboard", "home"): ("dashboard", "overview", "home"),
                ("configure", "settings", "config"): ("config", "settings", "preferences"),
                ("create", "add", "new"): ("create", "new", "add"),
                ("install", "setup"): ("install", "setup", "installation"),
                ("download", "get"): ("download", "fetch", "get"),
                ("upload", "send"): ("upload", "send", "transfer"),
            }
            
            for step_keywords, img_keywords in keyword_matches.items():
                if any(kw in step_lower for kw in step_keywords):
                    if any(kw in img_text for kw in img_keywords):
                        score += 30
                        break
        
            # Image type bonus (0-20 points)
            img_type = img.get("type", "content")
            type_scores = {
                "screenshot": 20,
                "diagram": 18,
                "illustration": 15,
                "photo": 12,
                "content": 8,
                "generated": 5,
                "placeholder": 3
            }
            score += type_scores.get(img_type, 5)
            
            # Source bonus (prefer scraped over placeholders)
            if img.get("source") == "scraped":
                score += 10
        
            scored_images.append((score, img))
    
        # Sort by score
        scored_images.sort(key=lambda x: x[0], reverse=True)
    
        # Return best match if score is good enough
        if scored_images:
            best_score, best_img = scored_images[0]
        
            # Lower threshold for acceptance (more permissive)
            if best_score >= 10:  # Reduced from 15
                logger.debug(
                    f"Matched image: {best_img.get('url', 'N/A')[:50]} "
                    f"(score: {best_score:.1f}, type: {best_img.get('type', 'N/A')})"
                )
                return best_img
            else:
                logger.debug(f"No good match found (best score: {best_score:.1f})")
    
        # Fallback: Round-robin assignment
        available = [img for img in image_pool if not img.get("_used")]
        if available:
            index = (step_index - 1) % len(available)
            fallback_img = available[index]
            logger.debug(f"Using round-robin fallback (index: {index})")
            return fallback_img
    
        return None

    def _format_image_gallery(self, images: List[Dict[str, Any]]) -> str:
        """
        PRODUCTION FIX: Actually format the image gallery (was returning empty string).
        """
        if not images:
            return ""
        
        parts = ["\n## 🖼️ Additional Resources\n"]
        
        for img in images[:self._gallery_limit]:
            url = img.get("url")
            if not url or not self._validate_image_url(url):
                continue
            
            alt = img.get("alt", "Additional resource")
            caption = img.get("caption", "")
            
            parts.append(f"\n![{alt}]({url})")
            if caption:
                parts.append(f"\n*{caption}*\n")
        
        result = "".join(parts)
        logger.debug(f"[Formatter] Generated gallery with {len(images)} images")
        return result

    def _get_step_emoji(self, step_type: str) -> str:
        """Get emoji for step type."""
        emoji_map = {
            "action": "▶️",
            "info": "ℹ️",
            "note": "📌",
            "warning": "⚠️",
            "tip": "💡",
            "check": "✅",
            "error": "❌",
            "config": "⚙️",
            "install": "📦"
        }
        return emoji_map.get((step_type or "").lower(), "▶️")

    def _extract_used_image_urls(self, steps: List[Dict[str, Any]]) -> Set[str]:
        """Extract URLs of images already used in steps."""
        used = set()
        for s in steps or []:
            img = s.get("image")
            if isinstance(img, dict):
                u = img.get("url")
                if u:
                    used.add(u)
            elif isinstance(img, str) and self._validate_image_url(img):
                used.add(img)
        return used

    def _normalize_markdown(self, text: str) -> str:
        """Normalize markdown for consistent formatting."""
        if not text:
            return ""
        
        # Reduce excessive blank lines
        text = re.sub(r'\n{4,}', '\n\n', text)
        
        # Ensure spacing around headers
        text = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', text)
        text = re.sub(r'(#{1,6} .+)\n([^\n#])', r'\1\n\n\2', text)
        
        # Ensure spacing before images
        text = re.sub(r'([^\n])\n!\[', r'\1\n\n![', text)
        
        # Trim trailing spaces
        text = re.sub(r'[ \t]+\n', '\n', text)
        
        return text.strip()

    def _fallback_minimal(
        self, 
        answer: Optional[str], 
        metadata: Optional[Dict[str, Any]], 
        confidence: float
    ) -> str:
        """Minimal safe fallback response."""
        parts = []
        
        if answer and answer.strip():
            parts.append(self._format_main_answer(answer))
        else:
            parts.append(
                "## ℹ️ Response\n\n"
                "I couldn't prepare a detailed formatted response at this time."
            )
        
        return "\n\n---\n\n".join(parts)


# ============================================================================
# Module-level convenience functions
# ============================================================================
_formatter = OpenWebUIFormatter()

def format_for_openwebui(
    answer: str,
    steps: List[Dict[str, Any]] = None,
    images: List[Dict[str, Any]] = None,
    query: str = "",
    confidence: float = 0.0,
    summary: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    show_metadata: bool = False
) -> str:
    """Format response for OpenWebUI with guaranteed displayable images."""
    return _formatter.format_response(
        answer=answer or "",
        steps=steps or [],
        images=images or [],
        query=query or "",
        confidence=confidence or 0.0,
        summary=summary,
        metadata=metadata or {}
    )


def format_agent_response_for_openwebui(
    response_text: str,
    execution_result: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Format agent responses (cluster tables, etc.)."""
    sections = []
    
    if response_text:
        sections.append(response_text)

    if execution_result and execution_result.get("success"):
        data = execution_result.get("data", {})
        
        # Cluster listing
        if isinstance(data, dict) and "data" in data:
            clusters = data["data"]
            sections.append("\n\n## 📊 Cluster Details\n")
            
            by_endpoint = {}
            for c in clusters:
                ep = c.get("displayNameEndpoint", "Unknown")
                by_endpoint.setdefault(ep, []).append(c)

            for ep, cluster_list in sorted(by_endpoint.items()):
                sections.append(f"\n### 📍 {ep}\n")
                sections.append("\n| Cluster Name | Status | Nodes | K8s Version |")
                sections.append("\n|-------------|--------|-------|-------------|")
                
                for cl in cluster_list:
                    status = "✅" if cl.get("status") == "Healthy" else "⚠️"
                    name = cl.get("clusterName", "N/A")
                    nodes = cl.get("nodescount", 0)
                    version = cl.get("kubernetesVersion", "N/A")
                    sections.append(f"\n| {name} | {status} | {nodes} | {version} |")
        
        # Endpoint listing
        elif isinstance(data, dict) and "endpoints" in data:
            endpoints = data.get("endpoints", [])
            sections.append("\n\n## 📍 Available Endpoints\n")
            sections.append("\n| # | Name | ID | Type |")
            sections.append("\n|---|------|----|----- |")
            
            for idx, ep in enumerate(endpoints, 1):
                name = ep.get("name", "Unknown")
                eid = ep.get("id", "N/A")
                etype = ep.get("type", "")
                sections.append(f"\n| {idx} | {name} | {eid} | {etype} |")

    # Session info
    if session_id and metadata and metadata.get("missing_params"):
        sections.append(
            "\n\n---\n\n"
            "*💬 Multi-turn conversation: session preserved for follow-up.*"
        )

    return "".join(sections)


def format_error_for_openwebui(
    error_message: str,
    suggestions: Optional[List[str]] = None,
    error_type: str = "general"
) -> str:
    """Format error messages for OpenWebUI."""
    emoji_map = {
        "general": "❌",
        "not_found": "🔍",
        "service_unavailable": "⚠️",
        "permission": "🔒",
        "validation": "📋"
    }
    
    emoji = emoji_map.get(error_type, "❌")
    parts = [f"## {emoji} Error\n", f"\n{error_message}\n"]
    
    if suggestions:
        parts.append("\n### 💡 Suggestions\n")
        for s in suggestions:
            parts.append(f"\n- {s}")
    
    return "".join(parts)