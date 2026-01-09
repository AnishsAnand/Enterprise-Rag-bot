
import logging
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenWebUIFormatter:
    """
    Production-grade formatter ensuring EVERY step has an associated image.
    """

    def __init__(self):
        self._gallery_limit = 8
        self._max_step_images = 20
        
        # Placeholder service for image prompts (when no real image exists)
        self.placeholder_service = "https://via.placeholder.com"

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
        Format complete response with GUARANTEED images for steps.
        """
        try:
            # Normalize inputs
            steps_list = self._normalize_steps(steps)
            images_list = self._normalize_images(images)

            logger.info(
                f"[Formatter] Processing: {len(steps_list)} steps, "
                f"{len(images_list)} images available"
            )

            # If no steps provided, infer from answer
            if not steps_list and answer:
                steps_list = self._infer_steps_from_text(answer)

            sections: List[str] = []

            # Summary section
            if summary and summary.strip() and summary.strip() != (answer or "").strip():
                sections.append(self._format_summary(summary))

            # Main answer section
            if answer and answer.strip():
                sections.append(self._format_main_answer(answer))

            # CRITICAL: Steps with GUARANTEED images
            if steps_list:
                formatted_steps = self._format_steps_with_guaranteed_images(
                    steps_list, 
                    images_list,
                    query
                )
                sections.append(formatted_steps)

            # Remaining images gallery (if any unused images)
            used_urls = self._extract_used_image_urls(steps_list)
            unused_images = [
                img for img in images_list 
                if img.get("url") and img.get("url") not in used_urls
            ]
            if unused_images:
                sections.append(
                    self._format_image_gallery(unused_images[:self._gallery_limit])
                )

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
                        "image": None
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
                        "image": None
                    })
                    
        except Exception as e:
            logger.debug(f"[Formatter] Step normalization error: {e}")
        
        return [s for s in normalized if s.get("text")]

    def _normalize_images(self, images) -> List[Dict[str, Any]]:
        """Normalize images to standard format with deduplication."""
        if not images:
            return []
            
        normalized: List[Dict[str, Any]] = []
        
        try:
            if isinstance(images, dict):
                images = [images]
                
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, str):
                        # String URL
                        if img.startswith("http"):
                            normalized.append({
                                "url": img,
                                "alt": "",
                                "caption": "",
                                "type": ""
                            })
                    elif isinstance(img, dict):
                        url = img.get("url") or img.get("src") or ""
                        
                        # Handle image prompts (no URL but has description)
                        if not url and img.get("image_prompt"):
                            normalized.append({
                                "image_prompt": img.get("image_prompt"),
                                "alt": img.get("alt", ""),
                                "caption": img.get("caption", ""),
                                "type": "prompt"
                            })
                        elif url:
                            normalized.append({
                                "url": url,
                                "alt": img.get("alt", "") or "",
                                "caption": img.get("caption", "") or "",
                                "type": img.get("type", ""),
                                "source_url": img.get("source_url", "")
                            })
                            
        except Exception as e:
            logger.debug(f"[Formatter] Image normalization error: {e}")
        
        # Deduplicate by URL
        seen: Set[str] = set()
        deduplicated: List[Dict[str, Any]] = []
        
        for img in normalized:
            url = img.get("url")
            if url:
                if url not in seen:
                    seen.add(url)
                    deduplicated.append(img)
            else:
                # Keep prompts even without URLs
                deduplicated.append(img)
        
        return deduplicated

    def _infer_steps_from_text(self, text: str, max_steps: int = 6) -> List[Dict[str, Any]]:
        """Infer steps from text when not explicitly provided."""
        if not text:
            return []
            
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        steps = []
        
        for i, sentence in enumerate(sentences[:max_steps]):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Truncate very long sentences
            if len(sentence) > 600:
                sentence = sentence[:600].rsplit(' ', 1)[0] + "..."
            
            steps.append({
                "step_number": i + 1,
                "text": sentence,
                "type": "info",
                "image": None
            })
        
        return steps

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
        Format steps with GUARANTEED images for EVERY step.
        
        Strategy:
        1. Use explicit step image if provided
        2. Assign available images round-robin
        3. Generate placeholder for steps without images
        """
        if not steps:
            return ""

        formatted_parts: List[str] = ["## 📝 Step-by-Step Instructions\n"]
        
        # Create image pool (only real URLs)
        image_pool = [
            img for img in all_images 
            if img.get("url") and not img.get("_used")
        ]
        
        logger.info(
            f"[Formatter] Formatting {len(steps)} steps with "
            f"{len(image_pool)} available images"
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

            # === CRITICAL: Guarantee image for EVERY step ===
            image_md = None
            
            # Strategy 1: Use explicit step image
            step_image = step.get("image")
            if step_image:
                try:
                    image_md = self._format_step_image(step_image, step_number)
                except Exception as e:
                    logger.debug(f"[Formatter] Step {step_number} image error: {e}")

            # Strategy 2: Use image prompt if provided
            if not image_md and step.get("image_prompt"):
                prompt = step.get("image_prompt")
                formatted_parts.append(f"\n> 💡 **Visual Guide:** {prompt}\n")
                # Generate placeholder for the prompt
                placeholder_url = self._generate_placeholder_url(
                    prompt, 
                    step_number
                )
                image_md = f"![Step {step_number} Visual Guide]({placeholder_url})"

            # Strategy 3: Assign from image pool (round-robin)
            if not image_md and image_pool:
                # Get next available image
                pool_idx = (idx - 1) % len(image_pool)
                img = image_pool[pool_idx]
                
                url = img.get("url")
                alt = img.get("alt") or f"Step {step_number} illustration"
                caption = img.get("caption", "")
                
                image_md = f"![{alt}]({url})"
                if caption:
                    image_md += f"\n*{caption}*"
                
                # Mark as used
                img["_used"] = True

            # Strategy 4: Generate contextual placeholder
            if not image_md:
                placeholder_url = self._generate_placeholder_url(
                    step_text[:80],
                    step_number
                )
                image_md = f"![Step {step_number} illustration]({placeholder_url})"
            
            # Add the image
            if image_md:
                formatted_parts.append(f"\n{image_md}\n")
                logger.debug(f"[Formatter] Step {step_number}: Image embedded")
            
            # Add notes if present
            note = step.get("note")
            if note:
                formatted_parts.append(f"\n> **Note:** {note}\n")

            # Separator between steps
            if idx < len(steps):
                formatted_parts.append("\n---\n")

        result = "".join(formatted_parts)
        logger.info(f"[Formatter] Formatted {len(steps)} steps with images")
        return result

    def _format_step_image(
        self, 
        image: Any, 
        step_number: int
    ) -> Optional[str]:
        """Format a single image for a step."""
        # Handle string URL
        if isinstance(image, str):
            if image.startswith("http"):
                return f"![Step {step_number} Illustration]({image})"
            else:
                # Treat as image prompt
                placeholder_url = self._generate_placeholder_url(image, step_number)
                return f"![Step {step_number} Visual Guide]({placeholder_url})"

        if not isinstance(image, dict):
            return None

        # Handle image prompt
        prompt = image.get("image_prompt")
        if prompt and not image.get("url"):
            placeholder_url = self._generate_placeholder_url(prompt, step_number)
            return f"![Step {step_number} Visual Guide]({placeholder_url})"

        # Handle URL
        url = image.get("url") or image.get("src") or ""
        if not url or not isinstance(url, str):
            return None

        alt = (image.get("alt") or f"Step {step_number} illustration").strip()
        caption = (image.get("caption") or "").strip()

        markdown = f"![{alt}]({url})"
        if caption:
            markdown += f"\n*{caption}*"
        
        return markdown

    def _generate_placeholder_url(
        self, 
        text: str, 
        step_number: int,
        size: tuple = (900, 400)
    ) -> str:
        """
        Generate placeholder image URL with text overlay.
        Uses via.placeholder.com for reliable placeholder generation.
        """
        try:
            # Clean and truncate text
            clean_text = text.strip()[:100]
            # Remove special characters that might break URL
            clean_text = re.sub(r'[^\w\s-]', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            
            # URL encode
            encoded = quote_plus(f"Step {step_number}: {clean_text}")
            
            width, height = size
            bg_color = "4A90E2"  # Professional blue
            text_color = "FFFFFF"  # White text
            
            return (
                f"{self.placeholder_service}/{width}x{height}/{bg_color}/{text_color}"
                f"?text={encoded}"
            )
        except Exception as e:
            logger.debug(f"[Formatter] Placeholder generation error: {e}")
            return f"{self.placeholder_service}/900x400.png?text=Step+{step_number}"

    def _format_image_gallery(self, images: List[Dict[str, Any]]) -> str:
        """Format remaining images as a gallery."""
        if not images:
            return ""
            
        gallery_parts: List[str] = ["## 🖼️ Additional Visual References\n"]
        
        for idx, img in enumerate(images[:self._gallery_limit], 1):
            url = img.get("url")
            if not url:
                # Handle image prompts in gallery
                prompt = img.get("image_prompt")
                if prompt:
                    gallery_parts.append(
                        f"\n### Visual Guide {idx}\n\n> 💡 {prompt}\n"
                    )
                continue
            
            alt = img.get("alt") or f"Reference image {idx}"
            caption = img.get("caption") or ""
            
            gallery_parts.append(f"\n### Image {idx}\n")
            gallery_parts.append(f"![{alt}]({url})\n")
            
            if caption:
                gallery_parts.append(f"*{caption}*\n")
        
        return "".join(gallery_parts) if len(gallery_parts) > 1 else ""

    def _get_step_emoji(self, step_type: str) -> str:
        """Get emoji for step type."""
        emoji_map = {
            "action": "▶️",
            "info": "ℹ️",
            "note": "📌",
            "warning": "⚠️",
            "tip": "💡",
            "check": "✅",
            "error": "❌"
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
            elif isinstance(img, str) and img.startswith("http"):
                used.add(img)
        return used

    def _normalize_markdown(self, text: str) -> str:
        """Normalize markdown for consistent formatting."""
        if not text:
            return ""
        
        # Reduce excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure spacing around headers
        text = re.sub(r'([^\n])\n(#{1,6} )', r'\1\n\n\2', text)
        text = re.sub(r'(#{1,6} .+)\n([^\n])', r'\1\n\n\2', text)
        
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
    """Format response for OpenWebUI with guaranteed images."""
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