from typing import List, Dict, Any, Optional, Tuple
import os
import re
import json
import openai
import time
import asyncio
import httpx
from datetime import datetime
import logging
from dotenv import load_dotenv
from functools import partial, lru_cache
from httpx import TimeoutException, ConnectError
import difflib
import hashlib
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from httpx import AsyncClient, Limits, Timeout

logger = logging.getLogger(__name__)
load_dotenv()

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "25"))
MAX_RETRIES = int(os.getenv("AI_SERVICE_MAX_RETRIES", "2"))  # Reduced from 3 to 2 for faster failure
RETRY_BACKOFF_BASE = float(os.getenv("AI_SERVICE_BACKOFF_BASE", "2.0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B").strip()
HOSTED_EMBEDDING_MODEL = os.getenv("HOSTED_EMBEDDING_MODEL", "openai/gpt-oss-20b-embedding")
MIN_RELEVANCE_THRESHOLD = float(os.getenv("MIN_RELEVANCE_THRESHOLD", "0.25"))
MAX_CHUNKS_RETURN = int(os.getenv("MAX_CHUNKS_RETURN", "12"))

REQUIRED_EMBEDDING_DIM = 4096
EMBEDDING_SIZE_FALLBACK = REQUIRED_EMBEDDING_DIM
PRIMARY_CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
# Fallback models disabled - they all return 500 errors
# Only use PRIMARY_CHAT_MODEL (openai/gpt-oss-120b) which is fast (0.2-0.3s) and reliable
FALLBACK_CHAT_MODELS = [
    "openai/gpt-3.5-turbo",
    "meta/llama-3.1-70b-instruct", 
    "meta/Llama-3.1-8B-Instruct",
    "openai/gpt-oss-120b",  # Moved to end as it's failing
]
# Previously: ["openai/gpt-oss-20b", "meta/llama-3.1-70b-instruct", "meta/Llama-3.1-8B-Instruct"]

GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://models.cloudservices.tatacommunications.com/v1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)

_env_embedding_dim = os.getenv("EMBEDDING_DIMENSION")
if _env_embedding_dim and int(_env_embedding_dim) != REQUIRED_EMBEDDING_DIM:
    logger.critical(
        f"❌ CRITICAL: EMBEDDING_DIMENSION mismatch! "
        f"Code requires {REQUIRED_EMBEDDING_DIM} but .env has {_env_embedding_dim}. "
        f"This WILL cause vector database errors!"
    )
    raise RuntimeError(
        f"Embedding dimension mismatch: code={REQUIRED_EMBEDDING_DIM}, env={_env_embedding_dim}"
    )

logger.info(f"✅ Embedding dimension validated: {REQUIRED_EMBEDDING_DIM}")


class AIServiceError(Exception):
    """Custom exception for AI service errors"""
    pass


def _exp_backoff_sleep(attempt: int):
    """Exponential backoff with jitter"""
    import random
    sleep_for = (RETRY_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.5)
    time.sleep(min(sleep_for, 30))


async def _async_exp_backoff_sleep(attempt: int):
    """Async exponential backoff with jitter"""
    import random
    sleep_for = (RETRY_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.5)
    await asyncio.sleep(min(sleep_for, 30))


class AIService:
    """
    Enhanced AI Service with automatic service connection,
    intelligent task routing, and adaptive response generation.
    """
    
    def __init__(self):
        self.grok_client: Optional[Any] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.is_healthy: bool = False
        self.last_error: Optional[str] = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 1000
        self.current_chat_model: str = PRIMARY_CHAT_MODEL
        self.http_pool = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(30.0)
        )
        
        # Service connection registry
        self.connected_services: Dict[str, bool] = {
            "embedding": False,
            "chat": False,
            "http_fallback": False
        }
        self.failed_models: set = set()  
        self.working_models: List[str] = []
        
        # User requirements storage
        self.user_requirements_buffer: List[Dict[str, Any]] = []
        self.max_requirements_buffer = 100
        
        # Task execution history
        self.task_history: List[Dict[str, Any]] = []
        
        self.setup_clients()

        # Enhanced response templates with quality markers
        self.response_templates = {
            "informational": (
                "Provide a comprehensive and accurate answer based on the context below. "
                "Focus on factual information and cite specific details from the sources. "
                "Structure your response logically with clear sections. "
                "If information is insufficient, clearly state what's missing."
            ),
            "instructional": (
                "Give clear, step-by-step instructions based on the information provided. "
                "Ensure accuracy and completeness. Number each step clearly. "
                "Include warnings or important notes where relevant. "
                "Verify all technical details against the provided context."
            ),
            "troubleshooting": (
                "Analyze the issue systematically and provide a structured solution with clear steps. "
                "Reference specific technical details from the context. "
                "Prioritize the most common solutions first. "
                "Include diagnostic steps if applicable."
            ),
            "explanatory": (
                "Explain the concept clearly with examples and practical applications. "
                "Use the provided context to ensure accuracy. "
                "Break down complex ideas into understandable parts. "
                "Include relevant analogies or comparisons when helpful."
            ),
        }

        self.image_styles = {
            "technical": "clean, professional technical diagram with clear labels and arrows",
            "instructional": "clear step-by-step visual guide with numbered steps and annotations",
            "conceptual": "modern clean illustration with good contrast and clear visual hierarchy",
            "troubleshooting": "problem-solution visual with before/after comparison and clear indicators",
        }

    async def close(self):
        """Cleanup resources"""
        if self.http_pool:
            await self.http_pool.aclose()
    

    def _validate_embedding_dimension(self, embeddings: List[List[float]], expected_dim: int = REQUIRED_EMBEDDING_DIM) -> None:
        """Validate embedding dimensions"""
        if not embeddings:
            return
        actual_dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], (list, tuple)) else 0
        if actual_dim != expected_dim:
            error_msg = (
                f"CRITICAL EMBEDDING DIMENSION MISMATCH! "
                f"Expected: {expected_dim}, Got: {actual_dim}. "
                f"Model: {EMBEDDING_MODEL}. "
                f"This indicates the embedding model is returning incorrect dimensions. "
                f"Database operations WILL FAIL. Check model configuration."
            )
            logger.critical(f"❌ {error_msg}")
            raise AIServiceError(error_msg)
    
        logger.debug(f"✅ Embedding dimension validated: {actual_dim} (expected: {expected_dim})")

    def setup_clients(self) -> None:
        """Initialize HTTP client and OpenAI-compatible Grok SDK client"""
        try:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
            timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS, connect=15.0, read=HTTP_TIMEOUT_SECONDS)
            self.http_client = httpx.AsyncClient(
                timeout=timeout, 
                limits=limits, 
                trust_env=True,
                follow_redirects=True
            )
            self.connected_services["http_fallback"] = True
            logger.info("✅ HTTP client initialized")
        except Exception as e:
            logger.error(f"❌ HTTP client setup failed: {e}")
            self.http_client = None
            self.connected_services["http_fallback"] = False

        grok_key = os.getenv("GROK_API_KEY")
        if not grok_key:
            logger.critical("❌ CRITICAL: GROK_API_KEY not found in environment variables")
            self.last_error = "GROK_API_KEY not configured"
            self.grok_client = None
            self.is_healthy = False
            return

        try:
            self.grok_client = openai.OpenAI(
                base_url=GROK_BASE_URL, 
                api_key=grok_key, 
                timeout=HTTP_TIMEOUT_SECONDS,
                max_retries=0
            )
            logger.info("✅ Grok client initialized")

            ok = self._test_grok_connection()
            if ok:
                logger.info(f"✅ Grok service is healthy and ready (using model: {self.current_chat_model})")
                self.is_healthy = True
                self.connected_services["chat"] = True
                self.connected_services["embedding"] = True
                self.last_error = None
            else:
                logger.warning("⚠️ Grok service connection test failed")
                self.is_healthy = False
                self.connected_services["chat"] = False
                self.connected_services["embedding"] = False
                if not self.last_error:
                    self.last_error = "Grok connection test failed"
        except Exception as e:
            logger.error(f"❌ Grok client setup failed: {e}")
            self.last_error = str(e)
            self.grok_client = None
            self.is_healthy = False
            self.connected_services["chat"] = False
            self.connected_services["embedding"] = False

    def _test_grok_connection(self) -> bool:
        if not self.grok_client:
            self.last_error = "Grok client not initialized"
            return False

        models_to_try = [PRIMARY_CHAT_MODEL] + FALLBACK_CHAT_MODELS

        for model in models_to_try:
        # Skip known failed models
            if model in self.failed_models:
                logger.debug(f"⏭️ Skipping failed model: {model}")
                continue
        
            try:
                logger.info(f"🔄 Testing model: {model}")
                resp = self.grok_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=15
                )
        
                if resp and (getattr(resp, "choices", None) or 
                        (isinstance(resp, dict) and resp.get("choices"))):
                    self.current_chat_model = model
                    if model not in self.working_models:
                        self.working_models.append(model)
                    logger.info(f"✅ Connected with model: {model}")
                    return True
            
            except openai.APIError as e:
                status_code = getattr(e, 'status_code', None)
                logger.error(f"❌ Model {model} failed: HTTP {status_code}")
            
            # Mark as failed and continue
                if status_code in [500, 502, 503, 404]:
                    self.failed_models.add(model)
                    continue
                elif status_code in [401, 403]:
                    return False  # Auth error is fatal
                
            except Exception as e:
                logger.error(f"❌ Model {model} error: {str(e)[:200]}")
                self.failed_models.add(model)
                continue

        self.last_error = f"All models failed: {', '.join(models_to_try)}"
        return False

    def _ensure_service_available(self) -> None:
        """Raise if Grok SDK client isn't available"""
        if not self.grok_client:
            error_msg = f"Grok service unavailable: {self.last_error or 'Not initialized'}"
            logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def auto_connect_services(self) -> Dict[str, bool]:
        """
        Automatically attempt to connect/reconnect to all services.
        Returns dictionary of service connection statuses.
        """
        logger.info("🔄 Auto-connecting services...")
        
        # Try to reconnect HTTP client if disconnected
        if not self.connected_services["http_fallback"]:
            try:
                if not self.http_client or self.http_client.is_closed:
                    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
                    timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS, connect=15.0, read=HTTP_TIMEOUT_SECONDS)
                    self.http_client = httpx.AsyncClient(
                        timeout=timeout,
                        limits=limits,
                        trust_env=True,
                        follow_redirects=True
                    )
                self.connected_services["http_fallback"] = True
                logger.info("✅ HTTP client reconnected")
            except Exception as e:
                logger.warning(f"⚠️ HTTP client reconnection failed: {e}")
                self.connected_services["http_fallback"] = False
        
        # Try to reconnect Grok client if disconnected
        if not self.connected_services["chat"] or not self.connected_services["embedding"]:
            try:
                grok_key = os.getenv("GROK_API_KEY")
                if grok_key:
                    if not self.grok_client:
                        self.grok_client = openai.OpenAI(
                            base_url=GROK_BASE_URL,
                            api_key=grok_key,
                            timeout=HTTP_TIMEOUT_SECONDS,
                            max_retries=0
                        )
                    
                    ok = self._test_grok_connection()
                    if ok:
                        self.connected_services["chat"] = True
                        self.connected_services["embedding"] = True
                        self.is_healthy = True
                        logger.info("✅ Grok services reconnected")
                    else:
                        logger.warning("⚠️ Grok reconnection test failed")
            except Exception as e:
                logger.warning(f"⚠️ Grok client reconnection failed: {e}")
                self.connected_services["chat"] = False
                self.connected_services["embedding"] = False
        
        connection_summary = {
            "timestamp": datetime.now().isoformat(),
            "services": self.connected_services.copy(),
            "overall_healthy": self.is_healthy,
            "current_model": self.current_chat_model if self.connected_services["chat"] else None
        }
        
        logger.info(f"🔗 Service connection status: {self.connected_services}")
        return connection_summary

    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Deep query analysis for better retrieval"""
    
        analysis_prompt = f"""Analyze this search query:
"{query}"

Provide:
1. Query type (factual/procedural/conceptual/comparative)
2. Key entities (extract specific names, versions, technologies)
3. Implicit requirements (what user really needs)
4. Suggested expansions (related terms to search)

Respond in JSON:
{{
  "type": "...",
  "entities": ["...", "..."],
  "implicit_needs": ["...", "..."],
  "expansions": ["...", "..."],
  "is_technical": true/false
}}"""
    
        response = await self._call_chat_with_retries(
        analysis_prompt,
        max_tokens=300,
        temperature=0.1
        )
    
        try:
            return json.loads(self._extract_json_safe(response))
        except:
            return {
            "type": "general",
            "entities": [],
            "implicit_needs": [query],
            "expansions": [],
            "is_technical": False
            }

    async def detect_task_intent(self, query: str) -> Dict[str, Any]:
        """
        Use AI to detect task intent from natural language queries.
        Returns structured intent information for orchestration.
        """
        if not self.connected_services["chat"]:
            await self.auto_connect_services()
        
        if not self.connected_services["chat"]:
            # Fallback to pattern matching
            return {
                "type": "unknown",
                "confidence": 0.0,
                "original_query": query,
                "error": "AI service unavailable for intent detection"
            }
        
        prompt = (
            f"Analyze the following user query and determine the primary task intent.\n\n"
            f"User Query: {query}\n\n"
            "Classify the intent into one of these categories:\n"
            "- scrape: User wants to extract data from a website/URL\n"
            "- search: User wants to find information from knowledge base\n"
            "- analyze: User wants analysis, explanation, or summary\n"
            "- upload: User wants to process/upload a file or document\n"
            "- bulk_operation: User wants to scrape multiple pages/URLs\n"
            "- unknown: Intent is unclear\n\n"
            "Respond ONLY with valid JSON in this format:\n"
            '{\n'
            '  "type": "category_name",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "extracted_params": ["param1", "param2"],\n'
            '  "reasoning": "brief explanation"\n'
            '}'
        )
        
        try:
            response = await self._call_chat_with_retries(
                prompt,
                max_tokens=200,
                temperature=0.1,
                timeout=15
            )
            
            # Try to extract JSON
            intent_data = self._extract_json_safe(response)
            if intent_data:
                intent_data["original_query"] = query
                return intent_data
            
        except Exception as e:
            logger.warning(f"AI intent detection failed: {e}")
        
        return {
            "type": "unknown",
            "confidence": 0.0,
            "original_query": query,
            "error": "Intent detection failed"
        }

    async def store_user_requirement(self, requirement: Dict[str, Any]):
        """
        Store user requirement when task cannot be completed.
        This helps track what users need but services can't provide.
        """
        requirement_entry = {
            "timestamp": datetime.now().isoformat(),
            "requirement": requirement,
            "services_status": self.connected_services.copy()
        }
        
        self.user_requirements_buffer.append(requirement_entry)
        
        # Keep buffer size manageable
        if len(self.user_requirements_buffer) > self.max_requirements_buffer:
            self.user_requirements_buffer = self.user_requirements_buffer[-self.max_requirements_buffer:]
        
        logger.info(f"📝 Stored user requirement: {requirement.get('type', 'unknown')} "
                   f"(buffer size: {len(self.user_requirements_buffer)})")

    async def get_stored_requirements(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get stored user requirements for analysis."""
        return self.user_requirements_buffer[-limit:]

    async def analyze_requirements_pattern(self) -> Dict[str, Any]:
        """
        Analyze patterns in stored requirements to identify common needs.
        """
        if not self.user_requirements_buffer:
            return {
                "total_requirements": 0,
                "patterns": [],
                "message": "No requirements stored yet"
            }
        
        # Count requirement types
        type_counts = {}
        missing_services = {}
        
        for entry in self.user_requirements_buffer:
            req = entry.get("requirement", {})
            req_type = req.get("type", "unknown")
            type_counts[req_type] = type_counts.get(req_type, 0) + 1
            
            # Track which services were missing
            services = entry.get("services_status", {})
            for service, available in services.items():
                if not available:
                    missing_services[service] = missing_services.get(service, 0) + 1
        
        patterns = [
            {
                "type": req_type,
                "count": count,
                "percentage": round((count / len(self.user_requirements_buffer)) * 100, 1)
            }
            for req_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "total_requirements": len(self.user_requirements_buffer),
            "patterns": patterns,
            "missing_services": missing_services,
            "analysis_timestamp": datetime.now().isoformat()
        }

    # ==================== Utilities ====================

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def strip_markdown(self, text: str) -> str:
        """Enhanced markdown removal"""
        if not text:
            return ""
        
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\-\*\+]\s*", "• ", text, flags=re.MULTILINE)
        text = re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        
        return text.strip()

    def _classify_query_type(self, query: str) -> str:
        """Enhanced query classification"""
        query_lower = (query or "").lower()
        
        if any(word in query_lower for word in [
            "how to", "steps", "guide", "tutorial", "instructions", 
            "procedure", "process", "way to", "method"
        ]):
            return "instructional"
        
        if any(word in query_lower for word in [
            "error", "problem", "issue", "fix", "troubleshoot", "debug", 
            "not working", "broken", "failed", "crash", "bug"
        ]):
            return "troubleshooting"
        
        if any(word in query_lower for word in [
            "what is", "explain", "define", "meaning", "concept", 
            "theory", "why", "understand", "difference between"
        ]):
            return "explanatory"
        
        return "informational"

    def _extract_message_content(self, choice_entry: Any) -> Optional[str]:
        """Safely extract content from various response shapes"""
        try:
            message_obj = getattr(choice_entry, "message", None)
            if message_obj is not None:
                content = getattr(message_obj, "content", None)
                if content:
                    return content
                if isinstance(message_obj, dict):
                    content = message_obj.get("content")
                    if content:
                        return content
            
            text_attr = getattr(choice_entry, "text", None)
            if text_attr:
                return text_attr
            
            if isinstance(choice_entry, dict):
                if "message" in choice_entry and isinstance(choice_entry["message"], dict):
                    return choice_entry["message"].get("content")
                if "text" in choice_entry:
                    return choice_entry.get("text")
        except Exception as e:
            logger.debug(f"Error extracting message content: {e}")
            return None
        
        return None

    def _extract_json_safe(self, raw: str) -> Optional[Dict[str, Any]]:
        """Safely extract JSON from text"""
        if not raw:
            return None
        
        # Try direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass
        
        # Try to find JSON object
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass
        
        return None

    # ==================== Embeddings ====================

    async def _generate_embeddings_http(self, texts: List[str], model: str) -> List[List[float]]:
        """HTTP fallback for embeddings with retry logic"""
        if not self.http_client:
            raise AIServiceError("HTTP client unavailable for embedding")

        grok_key = os.getenv("GROK_API_KEY")
        if not grok_key:
            raise AIServiceError("GROK_API_KEY not available")

        headers = {
            "Authorization": f"Bearer {grok_key}",
            "Content-Type": "application/json"
        }
        embeddings: List[List[float]] = []

        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
                continue

            last_exc: Optional[Exception] = None
            payload_text = text if text and text.strip() else " "
            
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    payload = {"input": payload_text, "model": model}
                    resp = await self.http_client.post(
                        f"{GROK_BASE_URL}/embeddings",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict) and data.get("data"):
                            emb = data["data"][0].get("embedding")
                            if emb and isinstance(emb, list) and len(emb) > 0:
                                if len(self._embedding_cache) < self._max_cache_size:
                                    self._embedding_cache[text_hash] = emb
                                embeddings.append(emb)
                                break
                            else:
                                last_exc = RuntimeError("Invalid embedding content")
                        else:
                            last_exc = RuntimeError(f"Invalid response: {resp.text[:200]}")
                    else:
                        last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                        
                except Exception as e:
                    last_exc = e
                    logger.debug(f"Embedding attempt {attempt} failed: {e}")

                if attempt < MAX_RETRIES:
                    await _async_exp_backoff_sleep(attempt)
                else:
                    logger.error(f"❌ HTTP embedding failed for index {i}: {last_exc}")
                    embeddings.append([0.0] * EMBEDDING_SIZE_FALLBACK)

        return embeddings

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with automatic service connection"""
        import time
        from app.services.prometheus_metrics import metrics
        
        start_time = time.time()
        
        if not texts:
            logger.warning("Empty texts provided for embedding generation")
            return []

        # Auto-connect if not connected
        if not self.connected_services["embedding"]:
            await self.auto_connect_services()

        cleaned_texts: List[str] = []
        for text in texts:
            if text is None:
                cleaned_texts.append(" ")
            elif isinstance(text, bytes):
                cleaned_texts.append(text.decode("utf-8", errors="replace")[:8000])
            else:
                s = str(text).strip()
                if not s:
                    s = " "
                cleaned_texts.append(s[:8000])

        # Try SDK path first
        if self.grok_client and self.connected_services["embedding"]:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    try:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                partial(
                                    self.grok_client.embeddings.create,
                                    input=cleaned_texts,
                                    model=EMBEDDING_MODEL
                                )
                            ),
                            timeout=HTTP_TIMEOUT_SECONDS
                        )
                        
                        data = getattr(result, "data", None) or (
                            result.get("data") if isinstance(result, dict) else None
                        )
                        
                        if data and len(data) == len(cleaned_texts):
                            embeddings: List[List[float]] = []
                            for entry in data:
                                if isinstance(entry, dict):
                                    emb = entry.get("embedding")
                                else:
                                    emb = getattr(entry, "embedding", None)
                                
                                if emb and isinstance(emb, list):
                                    embeddings.append(emb)
                                else:
                                    embeddings.append([0.0] * EMBEDDING_SIZE_FALLBACK)
                            
                            # Track successful embedding call
                            duration = time.time() - start_time
                            total_chars = sum(len(t) for t in cleaned_texts)
                            metrics.track_llm_call(
                                model=EMBEDDING_MODEL,
                                operation='embedding',
                                input_tokens=total_chars // 4,  # Approximate tokens
                                output_tokens=0,
                                duration=duration,
                                success=True
                            )
                            
                            logger.info(f"✅ Generated {len(embeddings)} embeddings via SDK (attempt {attempt})")
                            return embeddings
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Batch embedding timeout on attempt {attempt}")
                        raise
                    except Exception as batch_error:
                        logger.debug(f"Batch embedding failed: {batch_error}, trying per-item")
                        
                        embeddings = []
                        for t in cleaned_texts:
                            try:
                                res = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        partial(
                                            self.grok_client.embeddings.create,
                                            input=t,
                                            model=EMBEDDING_MODEL
                                        )
                                    ),
                                    timeout=30
                                )
                                
                                data = getattr(res, "data", None) or (
                                    res.get("data") if isinstance(res, dict) else None
                                )
                                
                                if data and len(data) > 0:
                                    entry = data[0]
                                    emb = entry.get("embedding") if isinstance(entry, dict) else getattr(entry, "embedding", None)
                                    if emb and isinstance(emb, list):
                                        embeddings.append(emb)
                                        continue
                            except Exception as item_error:
                                logger.debug(f"Per-item embedding failed: {item_error}")
                            
                            embeddings.append([0.0] * EMBEDDING_SIZE_FALLBACK)
                        
                        logger.info(f"✅ Generated {len(embeddings)} embeddings via SDK per-item (attempt {attempt})")
                        return embeddings
                        
                except Exception as e:
                    logger.warning(f"⚠️ SDK embedding attempt {attempt} failed: {e}")
                    if attempt < MAX_RETRIES:
                        await _async_exp_backoff_sleep(attempt)
                        continue
                    else:
                        logger.warning("⚠️ All SDK attempts failed, falling back to HTTP")
                        self.connected_services["embedding"] = False
                        break

        # HTTP fallback
        if self.http_client and self.connected_services["http_fallback"]:
            try:
                logger.info(f"Attempting HTTP embedding with model: {EMBEDDING_MODEL}")
                embeddings = await self._generate_embeddings_http(cleaned_texts, EMBEDDING_MODEL)
                
                all_zero = all(
                    len(e) == EMBEDDING_SIZE_FALLBACK and all(v == 0.0 for v in e)
                    for e in embeddings
                )
                
                if all_zero and HOSTED_EMBEDDING_MODEL:
                    logger.warning(f"⚠️ Primary model produced zeros, trying hosted fallback: {HOSTED_EMBEDDING_MODEL}")
                    hosted_embeddings = await self._generate_embeddings_http(cleaned_texts, HOSTED_EMBEDDING_MODEL)
                    return hosted_embeddings
                
                return embeddings
                
            except Exception as e:
                logger.error(f"HTTP embedding with primary model failed: {e}")
                if self.http_client and HOSTED_EMBEDDING_MODEL:
                    try:
                        logger.info(f"Attempting hosted fallback model: {HOSTED_EMBEDDING_MODEL}")
                        hosted_embeddings = await self._generate_embeddings_http(cleaned_texts, HOSTED_EMBEDDING_MODEL)
                        return hosted_embeddings
                    except Exception as e2:
                        logger.error(f"Hosted embedding fallback also failed: {e2}")

        # Store requirement if all failed
        await self.store_user_requirement({
            "type": "embedding_generation_failed",
            "texts_count": len(texts),
            "error": "All embedding generation methods failed"
        })
        
        raise AIServiceError("All embedding generation methods failed")

    # ==================== Chat Generation ====================

    def _llm_chat(self, prompt: str, max_tokens: int = 1500, 
              temperature: float = 0.2, system_message: str = None, 
              model: str = None) -> str:
        import time
        from app.services.prometheus_metrics import metrics
        
        start_time = time.time()
        self._ensure_service_available()
    
        if system_message is None:
            system_message = "You are a helpful AI assistant."

        messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
        ]

    # Use provided model or current working model
        model_to_use = model or self.current_chat_model
    
    # If current model failed, try working models
        if model_to_use in self.failed_models and self.working_models:
            model_to_use = self.working_models[0]
        logger.info(f"Switching to working model: {model_to_use}")
    
        try:
            resp = self.grok_client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            timeout=30
            )
        
            if hasattr(resp, "choices") and resp.choices:
                content = self._extract_message_content(resp.choices[0])
                if content:
                    # Track successful LLM call
                    duration = time.time() - start_time
                    input_tokens = getattr(resp.usage, 'prompt_tokens', 0) if hasattr(resp, 'usage') else len(prompt) // 4
                    output_tokens = getattr(resp.usage, 'completion_tokens', 0) if hasattr(resp, 'usage') else len(content) // 4
                    metrics.track_llm_call(
                        model=model_to_use,
                        operation='chat',
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        duration=duration,
                        success=True
                    )
                    return content
        
            logger.error(f"❌ Empty response from {model_to_use}")
            # Track failed call (empty response)
            duration = time.time() - start_time
            metrics.track_llm_call(
                model=model_to_use,
                operation='chat',
                duration=duration,
                success=False,
                error_type='empty_response'
            )
            return ""
        
        except Exception as e:
            err_str = str(e)
            logger.error(f"❌ Model {model_to_use} failed: {err_str[:300]}")
            
            # Track failed call
            duration = time.time() - start_time
            error_type = 'api_error'
            if "401" in err_str or "403" in err_str:
                error_type = 'auth_error'
            elif "500" in err_str or "502" in err_str:
                error_type = 'server_error'
            elif "timeout" in err_str.lower():
                error_type = 'timeout'
            elif "rate" in err_str.lower():
                error_type = 'rate_limit'
            
            metrics.track_llm_call(
                model=model_to_use,
                operation='chat',
                duration=duration,
                success=False,
                error_type=error_type
            )
        
        # Mark as failed
            self.failed_models.add(model_to_use)
        
        # Don't retry on certain errors
            if "401" in err_str or "403" in err_str:
                raise AIServiceError(f"Authentication error: {err_str}")
        
            if "500" in err_str or "502" in err_str:
                raise AIServiceError(f"Service unavailable: {err_str}")
        
            raise

    async def _call_chat_with_retries(self, prompt: str, max_tokens: int = 1500,
                                   temperature: float = 0.2, 
                                   system_message: str = None,
                                   timeout: float = None) -> str:
        if timeout is None:
            timeout = 30

    # Auto-connect if needed
        if not self.connected_services["chat"]:
            await self.auto_connect_services()

        if not self.connected_services["chat"]:
            raise AIServiceError("Chat service unavailable")

    # Reduce retries for faster failure
        MAX_ATTEMPTS = 2
    
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                fut = asyncio.to_thread(
                    partial(self._llm_chat, prompt, max_tokens, 
                        temperature, system_message)
                )
                result = await asyncio.wait_for(fut, timeout=timeout)
            
                if result and result.strip():
                    return result
            
                logger.warning(f"Empty response on attempt {attempt}")
                if attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise AIServiceError("Empty response")
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout after {timeout}s")
                raise AIServiceError(f"Timeout after {timeout}s")
            
            except AIServiceError:
                raise
            
            except Exception as e:
                logger.error(f"❌ Attempt {attempt} failed: {str(e)[:300]}")
                self.connected_services["chat"] = False
                raise AIServiceError(f"Service error: {str(e)[:200]}")

    
    async def answer_without_context(self, query: str) -> Dict[str, Any]:
  
        logger.info(f"📝 Generating response without context for: {query[:50]}...")
    
    # Try to generate answer using general knowledge
        try:
            prompt = (
                f"User Question: {query}\n\n"
                "Provide a helpful, accurate response based on general knowledge. "
                "Be clear and concise. If you're not certain, say so."
            )
        
            answer = await self._call_chat_with_retries(
                prompt,
             max_tokens=800,
                temperature=0.3,
                timeout=20  # Shorter timeout for no-context queries
            )
        
            if answer and len(answer.strip()) > 20:
            # Generate summary
                try:
                    summary = await self.generate_summary(answer, max_sentences=2, max_chars=300)
                except Exception:
                    summary = answer[:300] + "..." if len(answer) > 300 else answer
            
                return {
                "query": query,
                "answer": answer,
                "steps": [],
                "images": [],
                "sources": [],
                "has_sources": False,
                "confidence": 0.45,  # Lower confidence without sources
                "results_found": 0,
                "results_used": 0,
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "summaryTitle": "Quick Summary"
            }
    
        except Exception as e:
            logger.error(f"Failed to generate no-context response: {e}")
    
    # Final fallback
        return {
        "query": query,
        "answer": "I apologize, but I don't have enough information in my knowledge base to answer that question, and the AI service is currently unavailable. Please try again later or rephrase your question.",
        "steps": [],
        "images": [],
        "sources": [],
        "has_sources": False,
        "confidence": 0.0,
        "results_found": 0,
        "results_used": 0,
        "timestamp": datetime.now().isoformat(),
        "summary": "Service temporarily unavailable",
        "summaryTitle": "Status"
        }
            


    async def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response with automatic fallback"""
        try:
            enhanced = await self.generate_enhanced_response(query, context, query_type=None)
            if isinstance(enhanced, dict):
                return enhanced.get("text", "") or ""
            if isinstance(enhanced, str):
                return enhanced
            return ""
        except Exception as e:
            logger.error(f"generate_response failed: {e}")
            try:
                fallback_prompt = f"Question: {query}\n\nProvide a helpful answer based on general knowledge."
                raw = await self._call_chat_with_retries(
                    fallback_prompt,
                    max_tokens=800,
                    temperature=0.3,
                    timeout=30
                )
                return raw or "I'm unable to generate a response at this time."
            except Exception as inner:
                logger.error(f"Fallback also failed: {inner}")
                await self.store_user_requirement({
                    "type": "response_generation_failed",
                    "query": query,
                    "error": str(inner)
                })
                return "I apologize, but I'm unable to generate a response at this time due to a service error."

    async def generate_expanded_context(self, query: str, context: List[str]) -> str:
        """Generate expanded, enriched context from raw context"""
        if not context:
            logger.debug("No context provided for expansion")
            return ""

        limited_context = context[:8]
        context_text = "\n\n---\n\n".join(limited_context)
        
        if len(context_text) > 8000:
            sentences = context_text.split(". ")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) > 7500:
                    break
                truncated += sentence + ". "
            context_text = truncated or context_text[:8000]

        prompt = (
            f"Analyze the following context and create a comprehensive, accurate summary that directly relates to the user's query.\n\n"
            f"User Query: {query}\n\n"
            f"Context to analyze:\n{context_text}\n\n"
            "Instructions:\n"
            "1. Extract ALL key facts and information that directly answer or relate to the query\n"
            "2. Organize information logically with clear structure\n"
            "3. Maintain absolute accuracy - don't add information not present in the context\n"
            "4. Be comprehensive but concise - include technical details and specific examples\n"
            "5. Focus on actionable information when applicable\n"
            "6. Preserve important numbers, dates, and technical specifications\n"
            "7. If there are multiple perspectives or approaches, include them all\n\n"
            "Expanded Context:"
        )

        system_msg = (
            "You are an expert at analyzing and synthesizing information. "
            "Provide accurate, well-organized summaries based solely on the provided context. "
            "Never invent facts. Preserve technical accuracy and important details."
        )

        try:
            expanded = await self._call_chat_with_retries(
                prompt,
                max_tokens=1200,
                temperature=0.1,
                system_message=system_msg,
                timeout=HTTP_TIMEOUT_SECONDS
            )
            
            if expanded and len(expanded.strip()) > 50:
                return expanded
            else:
                logger.warning("Expanded context too short, using original")
                return context_text
                
        except AIServiceError as e:
            logger.error(f"Failed to generate expanded context: {e}")
            return context_text
        except Exception as e:
            logger.error(f"Unexpected error in context expansion: {e}")
            return context_text

    async def generate_enhanced_response(
    self, 
    query: str, 
    context: List[str], 
    query_type: str = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
        """PRODUCTION: Context-aware response with citation tracking"""
    
        if not query_type:
            query_type = self._classify_query_type(query)
    
    # Build rich context with source tracking
        context_with_sources = []
        for i, ctx in enumerate(context[:10], 1):
            if ctx and len(ctx.strip()) > 50:
                context_with_sources.append(f"[Source {i}]\n{ctx}\n")
    
        combined_context = "\n---\n".join(context_with_sources)
    
    # Enhanced prompt with explicit instructions
        template = self.response_templates.get(query_type, self.response_templates["informational"])
    
        prompt = f"""{template}

IMPORTANT INSTRUCTIONS:
1. Base your answer STRICTLY on the provided context
2. Cite sources using [Source N] notation when referencing specific information
3. If information is insufficient, clearly state what's missing
4. Provide specific examples and details from the context
5. Organize your response with clear structure

User Query: {query}

Available Context:
{combined_context}

Your Response (with source citations):"""
    
        system_msg = (
        f"You are an expert assistant specializing in {query_type} responses. "
        "Provide detailed, accurate, and well-structured information based "
        "strictly on the provided context. Always cite your sources using "
        "[Source N] notation. Never invent information not present in the context."
    )
    
        try:
            raw_response = await self._call_chat_with_retries(
            prompt=prompt,
            max_tokens=2500,
            temperature=temperature,
            system_message=system_msg,
            timeout=45
            )
        
            if not raw_response or len(raw_response.strip()) < 30:
                return {
                "text": "I apologize, but I couldn't generate a comprehensive response based on the available information.",
                "quality_score": 0.0,
                "query_type": query_type,
                "context_used": False,
                "expanded_context": combined_context[:500],
            }
        
        # Calculate quality based on multiple factors
            quality_score = self._calculate_response_quality_v2(
            response=raw_response,
            query=query,
            context=context,
            has_citations=bool(re.search(r'\[Source \d+\]', raw_response))
            )
        
            clean_response = self.strip_markdown(raw_response)
        
            return {
            "text": clean_response,
            "quality_score": quality_score,
            "query_type": query_type,
            "context_used": True,
            "expanded_context": combined_context,
            "has_citations": bool(re.search(r'\[Source \d+\]', raw_response)),
            }
        
        except Exception as e:
            logger.exception(f"Enhanced response generation failed: {e}")
            return {
            "text": "An error occurred while generating the response.",
            "quality_score": 0.0,
            "query_type": query_type,
            "context_used": False,
            "error": str(e)
            }
        
    def _calculate_response_quality_v2(self, response: str, query: str, context: List[str],has_citations: bool) -> float:
        """Enhanced quality scoring with citation bonus"""

        core: float = 0.0
    
        if not response or len(response.strip()) < 10:
            return 0.0
    
        score = 0.0
    
    # Length appropriateness (0-0.25)
        length = len(response)
        if 200 <= length <= 2000:
            score += 0.25
        elif 100 <= length < 200:
            score += 0.15
        elif length > 2000:
            score += 0.10
    
    # Query coverage (0-0.25)
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        coverage = len(query_terms & response_terms) / max(len(query_terms), 1)
        core += coverage * 0.25
    
    # Context usage (0-0.20)
        if context:
            context_text = " ".join(context).lower()
            context_terms = set(re.findall(r'\b\w+\b', context_text))
            context_usage = len(response_terms & context_terms) / max(len(context_terms), 1)
            score += context_usage * 0.20
    
    # Structure bonus (0-0.15)
        has_structure = bool(re.search(r'\n\n', response) or '. ' in response)
        if has_structure:
            score += 0.15
    
    # Citation bonus (0-0.15)
        if has_citations:
            score += 0.15
    
        return min(score, 1.0)
    
    async def generate_optimized_query_embedding(
        self,
        query: str,
        context_hints: Optional[List[str]] = None
    ) -> List[float]:
        """
        Generate optimized query embedding with expansion.
        
        Techniques:
        - Query expansion with synonyms
        - Context-aware embedding
        - Multi-vector representation
        """
        # Expand query with synonyms/related terms
        expanded_query = await self._expand_query(query, context_hints)
        
        # Generate embedding with expanded query
        embeddings = await self.generate_embeddings([expanded_query])
        
        return embeddings[0] if embeddings else []
    
    async def _expand_query(
        self,
        query: str,
        context_hints: Optional[List[str]] = None
    ) -> str:
        """
        Expand query with related terms using LLM.
        
        Example:
        "kubernetes cluster" -> "kubernetes cluster k8s pods containers orchestration"
        """
        expansion_prompt = f"""Given the search query: "{query}"

Generate 3-5 highly relevant related terms, synonyms, or alternative phrasings that would help find relevant documentation.

{f"Context: {', '.join(context_hints[:3])}" if context_hints else ""}

Respond with just the terms, comma-separated:"""
        
        try:
            response = await self._call_chat_with_retries(
                expansion_prompt,
                max_tokens=50,
                temperature=0.3,
                timeout=10
            )
            
            # Extract terms
            terms = [t.strip() for t in response.split(",") if t.strip()]
            
            # Combine with original query
            expanded = f"{query} {' '.join(terms[:5])}"
            
            logger.debug(f"Query expansion: '{query}' -> '{expanded}'")
            
            return expanded
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
    
    async def rerank_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using cross-encoder model or LLM.
        
        This significantly improves relevance compared to vector similarity alone.
        """
        if not results or len(results) <= top_k:
            return results
        
        # Use LLM for reranking (more accurate than pure vector similarity)
        reranked = await self._llm_rerank(query, results, top_k)
        
        return reranked
    
    async def _llm_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Use LLM to score and rerank results"""
        
        # Prepare results for scoring
        result_summaries = []
        for i, result in enumerate(results[:20]):  
            content = result.get("content", "")[:500]  
            result_summaries.append(f"[{i}] {content}")
        
        rerank_prompt = f"""Score these search results for relevance to the query: "{query}"

Results:
{chr(10).join(result_summaries)}

Rate each result 0-100 for relevance. Respond with ONLY comma-separated scores in order:
Example: 95,78,45,12,89,..."""
        
        try:
            response = await self._call_chat_with_retries(
                rerank_prompt,
                max_tokens=100,
                temperature=0.1,
                timeout=15
            )
            
            # Parse scores
            scores = []
            for score_str in response.split(","):
                try:
                    scores.append(float(score_str.strip()))
                except:
                    scores.append(0.0)
            
            # Combine with original results
            scored_results = []
            for i, result in enumerate(results[:len(scores)]):
                result_copy = result.copy()
                result_copy["rerank_score"] = scores[i] / 100.0  # Normalize to 0-1
                # Combine with original relevance score
                original_score = result.get("relevance_score", 0.5)
                result_copy["combined_score"] = (
                    0.7 * result_copy["rerank_score"] + 
                    0.3 * original_score
                )
                scored_results.append(result_copy)
            
            # Sort by combined score
            scored_results.sort(
                key=lambda x: x.get("combined_score", 0),
                reverse=True
            )
            
            logger.info(
                f"✅ Reranked {len(scored_results)} results "
                f"(top score: {scored_results[0].get('combined_score', 0):.2f})"
            )
            
            return scored_results[:top_k]
            
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            # Fallback to original order
            return results[:top_k]
    
    async def generate_domain_aware_response(
        self,
        query: str,
        context: List[str],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response with domain-specific formatting and terminology.
        
        Domains: technical, business, medical, legal, general
        """
        # Detect domain if not provided
        if not domain:
            domain = await self._detect_domain(query, context)
        
        # Get domain-specific system prompt
        system_prompt = self._get_domain_system_prompt(domain)
        
        # Generate response
        response = await self.generate_enhanced_response(
            query,
            context,
            query_type=None,
            temperature=0.2
        )
        
        return response
    
    async def _detect_domain(
        self,
        query: str,
        context: List[str]
    ) -> str:
        """Detect query domain using keywords"""
        
        combined_text = f"{query} {' '.join(context[:2])}"
        text_lower = combined_text.lower()
        
        domain_keywords = {
            "technical": [
                "api", "code", "function", "class", "method",
                "database", "server", "deploy", "configure",
                "kubernetes", "docker", "cluster", "endpoint"
            ],
            "business": [
                "revenue", "profit", "customer", "sales",
                "marketing", "strategy", "growth", "roi"
            ],
            "medical": [
                "patient", "diagnosis", "treatment", "symptom",
                "medication", "clinical", "therapy", "disease"
            ],
            "legal": [
                "contract", "agreement", "liability", "compliance",
                "regulation", "law", "clause", "jurisdiction"
            ]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            domain_scores[domain] = score
        
        # Get domain with highest score
        if domain_scores:
            detected = max(domain_scores.items(), key=lambda x: x[1])
            if detected[1] >= 2:  # At least 2 keyword matches
                return detected[0]
        
        return "general"
    
    def _get_domain_system_prompt(self, domain: str) -> str:
        """Get domain-specific system prompt"""
        
        prompts = {
            "technical": (
                "You are an expert technical documentation assistant. "
                "Provide precise, accurate technical information with proper "
                "code examples, command syntax, and configuration details. "
                "Use technical terminology correctly and include relevant "
                "warnings about version compatibility or deprecated features."
            ),
            "business": (
                "You are a business intelligence assistant. Provide clear, "
                "actionable insights with relevant metrics and data. Use "
                "business terminology appropriately and focus on practical "
                "implications and recommendations."
            ),
            "medical": (
                "You are a medical information assistant. Provide accurate, "
                "evidence-based information while emphasizing the importance "
                "of consulting healthcare professionals. Use proper medical "
                "terminology and cite sources when available."
            ),
            "legal": (
                "You are a legal information assistant. Provide accurate "
                "information about legal concepts and procedures while "
                "emphasizing that this is not legal advice and users should "
                "consult qualified legal professionals for specific situations."
            ),
            "general": (
                "You are a helpful, knowledgeable assistant. Provide clear, "
                "accurate information based on the provided context. Be "
                "thorough but concise, and structure your responses logically."
            )
        }
        
        return prompts.get(domain, prompts["general"])
    
    async def generate_stepwise_response(self, query: str, context: List[str],max_steps: int = 8) -> List[Dict[str, Any]]:

        try:
            organized_context = self._organize_context(context, query)
            candidate_images = self._extract_candidate_images_from_context(organized_context)
        
            available_images_json = json.dumps(
            candidate_images[:30],
            ensure_ascii=False,
            indent=2
            ) if candidate_images else "[]"
        
        # Step 3: Classify query type for better image assignment
            query_type = self._classify_query_type(query)
        
        # Step 4: Generate steps with AI
            logger.info(f"🔄 Generating stepwise response for: '{query[:60]}...'")
            steps = await self._generate_steps_with_ai_enhanced(
            query=query,
            context=organized_context,
            available_images_json=available_images_json,
            max_steps=max_steps,
            query_type=query_type
            )
        
        # Step 5: Validate and normalize steps
            validated_steps = self._validate_and_normalize_steps_enhanced(steps, max_steps)
        
        # Step 6: Assign images to steps intelligently
            final_steps = self._assign_images_to_steps(
            validated_steps,
            candidate_images,
            query_type
            )
        
        # Step 7: CRITICAL - Ensure all steps have image_prompt for OpenWebUI
            final_steps = self._ensure_image_prompts_for_openwebui(final_steps, query_type)
        
            logger.info(f"✅ Generated {len(final_steps)} steps with images for query")
            return final_steps
        
        except AIServiceError as e:
            logger.error(f"❌ AI Service error in stepwise generation: {e}")
            await self.store_user_requirement({
            "type": "stepwise_generation_failed",
            "query": query,
            "error": str(e)
            })
            return self._generate_emergency_fallback_steps(query, context, max_steps)
        
        except Exception as e:
            logger.exception(f"❌ Unexpected error in stepwise generation: {e}")
            return self._generate_emergency_fallback_steps(query, context, max_steps)


    async def _generate_steps_with_ai_enhanced(self,query: str,context: str,available_images_json: str,max_steps: int,query_type: str) -> List[Dict[str, Any]]:

    
        style_guidance = {
        "instructional": "step-by-step tutorial format with clear actions",
        "troubleshooting": "diagnostic and solution-focused approach",
        "explanatory": "educational format with clear explanations",
        "informational": "structured information delivery"
        }
    
        style = style_guidance.get(query_type, "clear and actionable format")
    
    # ✅ UPDATED PROMPT: No image_prompt requirement
        prompt = f"""Provide clear, actionable step-by-step instructions based on the context below.

Context:
{context}

Available Images (use ONLY if relevant):
{available_images_json}

User Question: {query}

INSTRUCTIONS:

1. Generate {max_steps} clear, specific steps in {style}

2. For images:
   - If a relevant image exists in Available Images, include it:
     "image": {{"url": "exact_url_from_above", "alt": "description", "caption": "optional"}}
   - If NO relevant image exists, OMIT the image field entirely
   - Do NOT create image_prompt or placeholder descriptions
   - Only include images that directly illustrate the step

3. Each step must be specific, clear, and actionable

4. Use appropriate step types:
   - "action": For steps requiring user action
   - "note": For warnings, tips, or cautions
   - "info": For context or background

5. Return ONLY valid JSON array (no markdown, no preamble):

[
  {{
    "step_number": 1,
    "text": "Clear step description",
    "type": "action",
    "image": {{"url": "https://...", "alt": "...", "caption": "..."}}
  }},
  {{
    "step_number": 2,
    "text": "Another step description",
    "type": "action"
  }}
]

IMPORTANT:
- Steps WITHOUT matching images should not have an "image" field at all
- Do NOT generate placeholder image descriptions
- Only include real image URLs from the Available Images list
"""

        try:
            raw_response = await self._call_chat_with_retries(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.15,
            timeout=30
            )
        
        # Parse JSON
            steps = self._extract_json_from_response_robust(raw_response)
        
            if not steps:
                logger.warning("⚠️ JSON extraction failed, trying text fallback")
                steps = self._parse_text_to_steps_fallback(raw_response, max_steps)
        
        # ✅ CRITICAL: Clean steps - remove any image_prompt fields
            cleaned_steps = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
            
            # Remove image_prompt if present
                if "image_prompt" in step:
                    del step["image_prompt"]
                    logger.debug(f"Removed image_prompt from step {step.get('step_number', '?')}")
            
            # Validate image field (must have real URL)
                if "image" in step:
                    img = step["image"]
                
                # If image is a dict, check URL
                    if isinstance(img, dict):
                        url = img.get("url", "")
                        if not url or not url.startswith("http"):
                        # Invalid URL - remove image field
                            del step["image"]
                            logger.debug(f"Removed invalid image from step {step.get('step_number', '?')}")
                
                # If image is a string, check if it's a URL
                    elif isinstance(img, str):
                        if not img.startswith("http"):
                        # Not a URL - remove image field
                            del step["image"]
                            logger.debug(f"Removed non-URL image from step {step.get('step_number', '?')}")
            
                cleaned_steps.append(step)
        
            logger.info(
            f"✅ Generated {len(cleaned_steps)} steps "
            f"(real images only, no placeholders)"
        )
            return cleaned_steps
        
        except Exception as e:
            logger.error(f"❌ AI step generation failed: {e}")
        # Return basic steps without images
            return self._create_fallback_steps_no_images(query, max_steps)
        
    def _create_fallback_steps_no_images(self, query: str, max_steps: int) -> List[Dict[str, Any]]:

        logger.warning("⚠️ Creating fallback steps (text-only, no images)")
    
        return [
            {
            "step_number": 1,
            "text": "Review the requirements and understand the task objective.",
            "type": "info"
            },
            {
            "step_number": 2,
            "text": "Gather necessary information and resources before proceeding.",
            "type": "action"
            },
            {
            "step_number": 3,
            "text": "Follow the documented procedure carefully, referring to official documentation.",
            "type": "action"
            },
            {
            "step_number": 4,
            "text": "Verify results and confirm successful completion.",
            "type": "action"
            }
        ][:max_steps]
        
    def _generate_auto_image_prompt(self, step_text: str, step_num: int, query_type: str) -> str:

        text_lower = step_text.lower()
    
    # Action-based prompts
        if "login" in text_lower or "sign in" in text_lower:
            return f"Screenshot of login interface showing username and password input fields with 'Sign In' button clearly visible. Highlight the username field."
    
        if "click" in text_lower or "select" in text_lower:
        # Try to extract what to click
            words = step_text.split()
            context = " ".join(words[:15])  # First 15 words for context
            return f"UI screenshot showing {context} with the clickable element circled in red with an arrow pointing to it."
    
        if "navigate" in text_lower or "go to" in text_lower:
            return f"Screenshot showing navigation menu or breadcrumb trail with the target location highlighted. Menu items should be clearly visible."
    
        if "configure" in text_lower or "settings" in text_lower:
            return f"Configuration panel screenshot showing relevant settings options with checkboxes, dropdowns, or input fields clearly labeled."
    
        if "enter" in text_lower or "type" in text_lower:
            return f"Form interface showing input field where data should be entered, with field label clearly visible and cursor positioned in the field."
    
        if "verify" in text_lower or "check" in text_lower:
            return f"Screenshot showing verification or confirmation screen with success indicators (checkmarks, green highlights) or validation results displayed."
    
        if "download" in text_lower or "upload" in text_lower:
            action = "download" if "download" in text_lower else "upload"
            return f"Screenshot of {action} interface showing file selection dialog, progress indicator, or {action} button prominently displayed."
    
        if "create" in text_lower or "add" in text_lower:
            return f"Screenshot of creation interface showing empty form or dialog with required fields, 'Create' or 'Add' button visible at bottom."
    
        if "delete" in text_lower or "remove" in text_lower:
            return f"Screenshot showing item selected for deletion with confirmation dialog asking 'Are you sure?' and Yes/No buttons."
    
    # Query type based prompts
        if query_type == "instructional":
            return f"Step-by-step visual guide showing: {step_text[:70]}, with numbered markers and arrows indicating sequence."
    
        if query_type == "troubleshooting":
            return f"Diagnostic screenshot highlighting the issue area with red border, and solution indicator with green checkmark showing correct state."
    
    # Generic fallback
        context = step_text[:80].strip()
        return f"Professional illustration demonstrating: {context}. Show clear labels, indicators, and visual hierarchy with appropriate colors and markers."
    
    def _create_fallback_steps_with_images(self, query: str, max_steps: int, query_type: str) -> List[Dict[str, Any]]:

        logger.warning(f"⚠️ Creating fallback steps with guaranteed image prompts")
    
        fallback_steps = [
        {
            "step_number": 1,
            "text": "Review the requirements and understand the task objective.",
            "type": "info",
            "image_prompt": "Flowchart or checklist showing the key requirements and objectives to accomplish, with checkboxes for each item."
        },
        {
            "step_number": 2,
            "text": "Gather all necessary information, credentials, and resources before proceeding.",
            "type": "action",
            "image_prompt": "Dashboard screenshot showing required resources panel with icons for credentials, documentation, and tools. Each item has a status indicator (ready/pending)."
        },
        {
            "step_number": 3,
            "text": "Follow the documented procedure carefully, referring to official documentation if needed.",
            "type": "action",
            "image_prompt": "Split-screen view showing documentation panel on left with step-by-step instructions, and implementation interface on right with current step highlighted."
        },
        {
            "step_number": 4,
            "text": "Verify the results and confirm successful completion of all steps.",
            "type": "action",
            "image_prompt": "Success confirmation screen showing green checkmarks for completed steps, with summary of results and 'Task Complete' banner at top."
        }
    ]
    
        return fallback_steps[:max_steps]

    def _extract_json_from_response_robust(self, response: str) -> List[Dict[str, Any]]:
    
    
        if not response or not response.strip():
            return []
    
    # Strategy 1: Direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict) and "steps" in data:
                return data["steps"] if isinstance(data["steps"], list) else []
        except json.JSONDecodeError:
            pass
    
    
        cleaned = re.sub(r'```(?:json)?\s*', '', response)
        cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Extract JSON array pattern using regex
        json_array_pattern = r'\[\s*\{.*?\}\s*\]'
        match = re.search(json_array_pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass
    
    # Strategy 4: Fix common JSON issues (trailing commas, etc.)
        try:
            fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            fixed = fixed.replace("'", '"')
            data = json.loads(fixed)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

        try:
            objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            steps = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and "text" in obj:
                        steps.append(obj)
                except Exception:
                    continue
            if steps:
                return steps
        except Exception:
            pass
    
        logger.warning("❌ All JSON extraction strategies failed")
        return []


    def _parse_text_to_steps_fallback(self, text: str, max_steps: int) -> List[Dict[str, Any]]:
    
        steps = []
        lines = text.split('\n')
    
    # Multiple patterns to catch different step formats
        step_patterns = [
        r'^(?:Step\s*)?(\d+)[\.\):\-]\s*(.+)',  # "Step 1." or "1." or "1:"
        r'^\s*[-*•]\s*(.+)',  # Bullet points
        r'^\s*\[(\d+)\]\s*(.+)',  # "[1] Step text"
        ]
    
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
        
            matched = False
            for pattern in step_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        step_num, text = match.groups()
                        if text and len(text.strip()) > 5:
                            steps.append({
                            "step_number": int(step_num) if step_num.isdigit() else len(steps) + 1,
                            "text": text.strip(),
                            "type": "action"
                            })
                            matched = True
                            break
                    else:
                        text = match.group(1)
                        if text and len(text.strip()) > 5:
                            steps.append({
                            "step_number": len(steps) + 1,
                            "text": text.strip(),
                            "type": "action"
                            })
                            matched = True
                            break
        
        # If no pattern matched but line is substantial, add it
            if not matched and len(line) > 20 and len(steps) < max_steps:
                steps.append({
                    "step_number": len(steps) + 1,
                    "text": line,
                    "type": "info"
                })
        
            if len(steps) >= max_steps:
                break
    
        return steps


    def _validate_and_normalize_steps_enhanced(self,steps: List[Dict[str, Any]],max_steps: int) -> List[Dict[str, Any]]:
        
        validated = []
    
        for idx, step in enumerate(steps[:max_steps], 1):
            if not isinstance(step, dict):
                logger.debug(f"Skipping non-dict step at index {idx}")
                continue
        
        # Extract and validate text
            text = step.get("text", "").strip()
            if not text or len(text) < 5:
                logger.debug(f"Skipping step {idx} with insufficient text")
                continue
        
        # Build normalized step
            normalized = {
            "step_number": step.get("step_number", idx),
            "text": text,
            "type": step.get("type", "action")
        }
        
        # Handle image field (URL-based) - multiple possible formats
            if "image" in step:
                if isinstance(step["image"], dict) and "url" in step["image"]:
                    normalized["image"] = {
                    "url": step["image"]["url"],
                    "alt": step["image"].get("alt", "Step illustration"),
                    "caption": step["image"].get("caption", "")
                }
                elif isinstance(step["image"], str) and step["image"].startswith("http"):
                    normalized["image"] = {
                        "url": step["image"],
                        "alt": "Step illustration",
                        "caption": ""
                    }
            elif "image_url" in step and isinstance(step["image_url"], str):
                normalized["image"] = {
                    "url": step["image_url"],
                    "alt": "Step illustration",
                    "caption": ""
                }
        
        # Handle image_prompt field
            if "image_prompt" in step and isinstance(step["image_prompt"], str):
                prompt = step["image_prompt"].strip()
                if prompt:
                    normalized["image_prompt"] = prompt
        
            validated.append(normalized)
    
    # Ensure at least one step exists
        if not validated:
            validated.append({
            "step_number": 1,
            "text": "Please refer to the provided context for detailed information.",
            "type": "info"
            })
    
        return validated


    def _ensure_image_prompts_for_openwebui(self, steps: List[Dict[str, Any]],query_type: str) -> List[Dict[str, Any]]:

        for step in steps:
        # Skip if already has image_prompt
            if "image_prompt" in step and step["image_prompt"]:
                continue
        
            text = step.get("text", "")
            step_type = step.get("type", "action")
            text_lower = text.lower()
        
        # Get appropriate style for this query type
            style = self.image_styles.get(query_type, self.image_styles["instructional"])
        
        # Generate contextual image prompt based on content keywords
            if "login" in text_lower or "sign in" in text_lower:
                prompt = f"Screenshot of login interface with username and password fields clearly visible. {style}"
        
            elif any(word in text_lower for word in ["click", "select", "choose", "press", "tap"]):
            # Extract what to click if possible
                button_match = re.search(r'(?:click|select|choose|press|tap)\s+(?:the\s+)?["\']?([^"\'.,]+)["\']?', text_lower)
                target = button_match.group(1) if button_match else "the button"
                prompt = f"UI screenshot with '{target}' highlighted and clearly indicated. {style}"
        
            elif "navigate" in text_lower or "go to" in text_lower or "open" in text_lower:
                location_match = re.search(r'(?:navigate to|go to|open)\s+(?:the\s+)?["\']?([^"\'.,]+)["\']?', text_lower)
                location = location_match.group(1) if location_match else "the target location"
                prompt = f"Screenshot showing navigation path to '{location}' with menu items visible. {style}"
        
            elif "configure" in text_lower or "setting" in text_lower or "option" in text_lower:
                prompt = f"Configuration panel screenshot showing relevant settings and options. {style}"
        
            elif "verify" in text_lower or "check" in text_lower or "confirm" in text_lower:
                prompt = f"Screenshot showing verification or confirmation screen with results visible. {style}"
        
            elif "download" in text_lower or "upload" in text_lower:
                action = "download" if "download" in text_lower else "upload"
                prompt = f"Screenshot showing {action} interface with file selection dialog. {style}"
        
            elif "create" in text_lower or "add" in text_lower or "new" in text_lower:
                prompt = f"Screenshot showing creation or addition interface with relevant fields. {style}"
        
            elif "delete" in text_lower or "remove" in text_lower:
                rompt = f"Screenshot showing deletion confirmation dialog or remove option. {style}"
        
            elif "search" in text_lower or "find" in text_lower:
                prompt = f"Screenshot showing search interface with search box and filters visible. {style}"
        
            elif "connect" in text_lower or "link" in text_lower or "integrate" in text_lower:
                prompt = f"Diagram showing connection or integration process with clear flow arrows. {style}"
        
            elif "error" in text_lower or "warning" in text_lower or "alert" in text_lower:
                prompt = f"Screenshot showing error message or warning dialog with details visible. {style}"
        
            elif "dashboard" in text_lower or "overview" in text_lower:
                prompt = f"Screenshot of dashboard or overview page with key metrics visible. {style}"
        
            elif "report" in text_lower or "export" in text_lower:
                prompt = f"Screenshot showing report generation or export interface. {style}"
        
            elif step_type == "note":
                prompt = f"Warning or information icon with important notice highlighted. {style}"
        
            elif step_type == "info":
                prompt = f"Informational diagram or illustration explaining the concept. {style}"
        
            else:
            # Generic prompt based on the step text (first 80 chars)
                context = text[:80].rstrip('.,!?')
                prompt = f"Visual guide illustrating: {context}. {style}"
        
        # Ensure prompt is not too long (max 200 chars for practicality)
            if len(prompt) > 200:
                prompt = prompt[:197] + "..."
        
            step["image_prompt"] = prompt
    
        return steps


    def _generate_emergency_fallback_steps(self,query: str,context: List[str],max_steps: int) -> List[Dict[str, Any]]:
    
    
        logger.warning("⚠️ Using emergency fallback step generation")
    
        steps = []
    
    # Try to extract sentences from context
        if context:
            text_source = " ".join(context[:2])
            sentences = [s.strip() for s in text_source.split(".") if len(s.strip()) > 10]
        
            for i, sentence in enumerate(sentences[:max_steps], 1):
            # Generate contextual image prompt
                sentence_lower = sentence.lower()
                if "login" in sentence_lower:
                    img_prompt = "Screenshot showing login interface"
                elif any(word in sentence_lower for word in ["click", "select"]):
                    img_prompt = "UI screenshot highlighting the relevant button"
                elif "navigate" in sentence_lower:
                    img_prompt = "Screenshot showing navigation menu"
                else:
                    img_prompt = f"Visual guide for: {sentence[:60]}"
            
                steps.append({
                "step_number": i,
                "text": sentence + ".",
                "type": "info",
                "image_prompt": img_prompt
            })
    
    # If no context or too few steps, create from query
        if not steps:
            query_words = query.split()
            if len(query_words) > 5:
                steps.append({
                "step_number": 1,
                "text": f"Review the requirements: {query}",
                "type": "info",
                "image_prompt": f"Overview diagram for: {query[:60]}"
            })
                steps.append({
                "step_number": 2,
                "text": "Gather necessary information and resources.",
                "type": "action",
                "image_prompt": "Checklist illustration showing required items"
                })
                steps.append({
                "step_number": 3,
                "text": "Follow the documented procedure carefully.",
                "type": "action",
                "image_prompt": "Step-by-step workflow diagram"
                })
            else:
                steps.append({
                "step_number": 1,
                "text": f"To accomplish: {query}",
                "type": "info",
                "image_prompt": f"Illustration for: {query[:60]}"
            })
    
    # Ensure we have at least one step
        if not steps:
            steps.append({
            "step_number": 1,
            "text": "Refer to the documentation for detailed instructions.",
            "type": "info",
            "image_prompt": "Documentation reference icon with book illustration"
            })
    
        return steps[:max_steps]

    async def generate_summary(self, text: str, max_sentences: int = 3, max_chars: int = 600) -> str:
        """Generate concise, informative summary"""
        if not text or len(text.strip()) < 50:
            return text[:max_chars] if text else ""

        if len(text) > 10000:
            sentences = text.split(". ")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) > 9000:
                    break
                truncated += sentence + ". "
            text = truncated or text[:10000]

        prompt = (
            f"Create a concise, informative summary of the following content.\n\n"
            f"Requirements:\n"
            f"- Maximum {max_sentences} sentences\n"
            f"- Focus on key points and main ideas\n"
            f"- Use clear, professional language\n"
            f"- Avoid repetition and filler words\n"
            f"- Include specific details and facts when relevant\n"
            f"- Make it actionable and useful\n\n"
            f"Content:\n{text}\n\n"
            f"Summary:"
        )

        system_msg = (
            "You are an expert at creating clear, concise summaries that capture essential information. "
            "Focus on accuracy and utility. Never invent information."
        )

        try:
            raw = await self._call_chat_with_retries(
                prompt,
                max_tokens=500,
                temperature=0.2,
                system_message=system_msg,
                timeout=30
            )
            
            if raw:
                summary = self.strip_markdown(raw).strip()
                
                if len(summary) > len(text):
                    summary = text[:max_chars]
                elif len(summary) < 20:
                    summary = text[:max_chars]
                
                return summary
                
        except (AIServiceError, Exception) as e:
            logger.error(f"Summary generation failed: {e}")

        sentences = text.split(". ")
        summary_parts = []
        current_length = 0
        
        for sentence in sentences[:max_sentences * 2]:
            if current_length + len(sentence) > max_chars:
                break
            summary_parts.append(sentence)
            current_length += len(sentence)
        
        return ". ".join(summary_parts) + "." if summary_parts else text[:max_chars]

    # ==================== Helper Methods ====================

    def _extract_json_array_safe(self, raw: str) -> List[Dict[str, Any]]:
        """Robust JSON array extraction"""
        if not raw:
            return []

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                if "steps" in data and isinstance(data["steps"], list):
                    return [item for item in data["steps"] if isinstance(item, dict)]
                return [data]
        except Exception:
            pass

        try:
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                data = json.loads(json_match.group(0))
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
        except Exception:
            pass

        try:
            objects = re.findall(r'\{[\s\S]*?\}', raw)
            steps = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and "text" in obj:
                        steps.append(obj)
                except Exception:
                    continue
            if steps:
                return steps
        except Exception:
            pass

        steps: List[Dict[str, Any]] = []
        lines = raw.split("\n")
        
        patterns = [
            r'^\s*(?:\d+[\.\)]\s*)(.+)',
            r'^\s*[-*•]\s*(.+)',
            r'^\s*Step\s*\d+\s*:\s*(.+)',
            r'^\s*\[?\d+\]?\s*(.+)',
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            matched = False
            for pattern in patterns:
                m = re.match(pattern, line, re.IGNORECASE)
                if m:
                    text = m.group(1).strip()
                    if len(text) > 5:
                        steps.append({
                            "text": text,
                            "type": "action"
                        })
                        matched = True
                        break
            
            if not matched and len(line) > 15:
                steps.append({
                    "text": line,
                    "type": "action"
                })
            
            if len(steps) >= 15:
                break
        
        return steps[:12]

    def _organize_context(self, context: List[str], query: str) -> str:
        """Organize context by relevance"""
        if not context:
            return "No specific context provided."

        query_terms = set(re.findall(r'\b\w+\b', (query or "").lower()))
        
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'get', 'has', 'how'
        }
        query_terms = query_terms - stopwords

        def relevance_score(text: str) -> float:
            if not text:
                return 0.0
            
            text_lower = text.lower()
            text_terms = set(re.findall(r'\b\w+\b', text_lower))
            
            term_overlap = len(query_terms.intersection(text_terms))
            length_score = min(len(text) / 2000.0, 1.0)
            phrase_matches = sum(1 for term in query_terms if term in text_lower)
            
            position_score = 0.0
            for term in query_terms:
                pos = text_lower.find(term)
                if pos != -1:
                    position_score += max(0, 1.0 - (pos / len(text_lower)))
            
            return (
                term_overlap * 3.0 +
                phrase_matches * 2.0 +
                length_score +
                position_score * 0.5
            )

        sorted_context = sorted(context[:12], key=relevance_score, reverse=True)
        
        organized = []
        for i, ctx in enumerate(sorted_context, 1):
            if ctx and len(ctx.strip()) > 30:
                if len(ctx) > 3000:
                    ctx = ctx[:3000] + "... [truncated]"
                organized.append(f"Source {i}:\n{ctx.strip()}")
        
        return "\n\n".join(organized) if organized else "Limited relevant context available."

    def _calculate_response_quality(self, response: str, query: str, context: List[str]) -> float:
        """Enhanced response quality calculation"""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        score = 0.2
        
        length = len(response)
        if 300 <= length <= 2500:
            score += 0.25
        elif 150 <= length < 300:
            score += 0.15
        elif 2500 < length <= 4000:
            score += 0.20
        elif length > 4000:
            score += 0.10

        query_terms = set(re.findall(r'\b\w+\b', (query or "").lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        coverage = len(query_terms.intersection(response_terms)) / max(len(query_terms), 1)
        score += coverage * 0.25

        if "\n\n" in response or ". " in response:
            paragraph_count = response.count("\n\n") + 1
            if 2 <= paragraph_count <= 6:
                score += 0.15
            elif paragraph_count > 1:
                score += 0.10

        if context:
            context_text = " ".join(context).lower()
            context_terms = set(re.findall(r'\b\w+\b', context_text))
            context_usage = len(response_terms.intersection(context_terms)) / max(len(context_terms), 1)
            score += context_usage * 0.15

        return min(score, 1.0)

    def _extract_candidate_images_from_context(self, context_text: str) -> List[Dict[str, Any]]:
        """Extract candidate images from context"""
        candidates: List[Dict[str, Any]] = []

        try:
            for m in re.finditer(
                r'!\[([^\]]*)\]\((https?://[^\)]+\.(?:png|jpe?g|gif|svg|webp|bmp)(?:\?[^\)]*)?)\)',
                context_text,
                flags=re.IGNORECASE
            ):
                alt = m.group(1).strip()
                url = m.group(2).strip()
                start, end = max(0, m.start() - 150), min(len(context_text), m.end() + 150)
                excerpt = context_text[start:end].strip().replace("\n", " ")
                candidates.append({
                    "url": url,
                    "alt": alt,
                    "caption": "",
                    "text": excerpt
                })

            for m in re.finditer(
                r'<img[^>]+src=["\'](https?://[^"\']+)["\'][^>]*>',
                context_text,
                flags=re.IGNORECASE
            ):
                url = m.group(1).strip()
                tag = m.group(0)
                alt_match = re.search(r'alt=["\']([^"\']+)["\']', tag, flags=re.IGNORECASE)
                alt = alt_match.group(1).strip() if alt_match else ""
                start, end = max(0, m.start() - 150), min(len(context_text), m.end() + 150)
                excerpt = context_text[start:end].strip().replace("\n", " ")
                candidates.append({
                    "url": url,
                    "alt": alt,
                    "caption": "",
                    "text": excerpt
                })

            for m in re.finditer(
                r'(https?://\S+\.(?:png|jpe?g|gif|svg|webp|bmp)(?:\?\S*)?)',
                context_text,
                flags=re.IGNORECASE
            ):
                url = m.group(1).strip().rstrip('),.;')
                start, end = max(0, m.start() - 150), min(len(context_text), m.end() + 150)
                excerpt = context_text[start:end].strip().replace("\n", " ")
                if not any(c["url"] == url for c in candidates):
                    candidates.append({
                        "url": url,
                        "alt": "",
                        "caption": "",
                        "text": excerpt
                    })

        except Exception as e:
            logger.debug(f"Error extracting candidate images: {e}")

        unique = []
        seen = set()
        for c in candidates:
            if c["url"] not in seen:
                seen.add(c["url"])
                unique.append(c)
        
        return unique

    def _score_text_similarity(self, a: str, b: str) -> float:
        """Enhanced text similarity scoring"""
        if not a or not b:
            return 0.0
        
        a_norm = " ".join(a.lower().split())
        b_norm = " ".join(b.lower().split())
        
        seq = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
        
        a_words = set(a_norm.split())
        b_words = set(b_norm.split())
        overlap = len(a_words & b_words) / (len(a_words | b_words) or 1)
        
        substring_bonus = 0.0
        if len(a_norm) > 10 and len(b_norm) > 10:
            if a_norm in b_norm or b_norm in a_norm:
                substring_bonus = 0.2
        
        return min(1.0, 0.5 * seq + 0.4 * overlap + 0.1 * substring_bonus)

    def _assign_images_to_steps(self, steps: List[Dict[str, Any]], 
                                candidate_images: List[Dict[str, Any]],
                                query_type: str) -> List[Dict[str, Any]]:
        """Intelligently assign images to steps"""
        enhanced = []
        used_image_indices = set()
        
        for i, step in enumerate(steps):
            s_text = step.get("text", "") or ""
            s_type = step.get("type", "action")
            step_img = step.get("image") or step.get("image_url") or step.get("image_prompt")

            assigned = None

            if isinstance(step_img, dict) and step_img.get("url"):
                assigned = {
                    "url": step_img.get("url"),
                    "alt": step_img.get("alt", "") or "",
                    "caption": step_img.get("caption", "") or "",
                }
            elif isinstance(step_img, str) and step_img.startswith("http"):
                assigned = {"url": step_img, "alt": "", "caption": ""}

            if not assigned and candidate_images:
                best_score = 0.0
                best_img = None
                best_idx = None
                
                for idx, img in enumerate(candidate_images):
                    if idx in used_image_indices:
                        continue
                    
                    text_blob = " ".join([
                        img.get("alt", ""),
                        img.get("caption", ""),
                        img.get("text", "")
                    ])
                    
                    score = self._score_text_similarity(s_text, text_blob)
                    
                    url_lower = (img.get("url") or "").lower()
                    if any(k in url_lower for k in [
                        "diagram", "chart", "screenshot", "flow", "graph",
                        "architecture", "schematic", "blueprint"
                    ]):
                        score += 0.08
                    
                    if f"step{i+1}" in url_lower or f"step-{i+1}" in url_lower:
                        score += 0.15
                    
                    if score > best_score:
                        best_score = score
                        best_img = img
                        best_idx = idx
                
                if best_img and best_score >= 0.15:
                    assigned = {
                        "url": best_img.get("url"),
                        "alt": best_img.get("alt", ""),
                        "caption": best_img.get("caption", "")
                    }
                    if best_idx is not None:
                        used_image_indices.add(best_idx)

            if not assigned:
                lp = step.get("image_prompt") or step.get("image_desc") or step.get("image_description")
                if lp and isinstance(lp, str) and lp.strip():
                    assigned = {"image_prompt": lp.strip()}
                else:
                    style = self.image_styles.get(query_type, self.image_styles["instructional"])
                    prompt = (
                        f"Create a clear, professional illustration for: {s_text.strip()}. "
                        f"Style: {style}. Focus on clarity, proper labeling, and visual hierarchy. "
                        f"The image should be self-explanatory and directly support the step description."
                    )
                    assigned = {"image_prompt": prompt}

            step_out = {
                "text": s_text,
                "type": s_type,
                "step_number": step.get("step_number", i + 1)
            }
            
            if assigned:
                step_out["image"] = assigned

            enhanced.append(step_out)
        
        return enhanced

    # ==================== Health Check ====================

    async def get_service_health(self) -> Dict[str, Any]:
        """Comprehensive service health check with auto-reconnection"""
        health_status: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "service": {
                "name": "grok",
                "available": bool(self.grok_client),
                "healthy": self.is_healthy,
                "status": "unknown",
                "last_error": self.last_error,
                "models": {
                    "chat": self.current_chat_model,
                    "chat_fallbacks": FALLBACK_CHAT_MODELS,
                    "embeddings": EMBEDDING_MODEL
                },
                "cache_size": len(self._embedding_cache),
                "connected_services": self.connected_services.copy(),
            },
            "overall_status": "unknown",
        }

        # Attempt auto-reconnection if unhealthy
        if not self.is_healthy:
            logger.info("🔄 Attempting auto-reconnection during health check...")
            await self.auto_connect_services()

        if self.grok_client:
            try:
                ok = self._test_grok_connection()
                if ok:
                    health_status["service"]["status"] = "healthy"
                    health_status["service"]["healthy"] = True
                    health_status["overall_status"] = "healthy"
                    health_status["service"]["current_model"] = self.current_chat_model
                    self.is_healthy = True
                    self.last_error = None
                    logger.info(f"✅ Service health check: HEALTHY (model: {self.current_chat_model})")
                else:
                    health_status["service"]["status"] = "unhealthy"
                    health_status["service"]["healthy"] = False
                    health_status["overall_status"] = "unhealthy"
                    health_status["service"]["last_error"] = self.last_error
                    logger.warning(f"⚠️ Service health check: UNHEALTHY - {self.last_error}")
            except Exception as e:
                health_status["service"]["status"] = "error"
                health_status["service"]["healthy"] = False
                health_status["overall_status"] = "error"
                health_status["error"] = str(e)
                self.last_error = str(e)
                logger.error(f"❌ Service health check error: {e}")
        else:
            health_status["overall_status"] = "unavailable"
            health_status["service"]["last_error"] = self.last_error or "Service not initialized"
            logger.error("❌ Grok client not initialized")

        # Add requirements analysis
        health_status["user_requirements"] = {
            "stored_count": len(self.user_requirements_buffer),
            "analysis": await self.analyze_requirements_pattern() if self.user_requirements_buffer else None
        }

        logger.info(f"Service health check completed - Status: {health_status['overall_status']}")
        return health_status

    # ==================== Cleanup ====================

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            try:
                await self.http_client.aclose()
                logger.info("✅ HTTP client closed successfully")
            except Exception as e:
                logger.warning(f"Failed to close HTTP client: {e}")

        if self.grok_client:
            try:
                if hasattr(self.grok_client, "aclose"):
                    await self.grok_client.aclose()
                elif hasattr(self.grok_client, "close"):
                    self.grok_client.close()
                logger.info("✅ Grok client closed successfully")
            except Exception as e:
                logger.debug(f"Grok client close attempt raised: {e}")
    
    def diagnose_connection(self) -> Dict[str, Any]:
        grok_key = os.getenv("GROK_API_KEY")
    
        return {
        "api_key_present": bool(grok_key),
        "api_key_length": len(grok_key) if grok_key else 0,
        "api_key_preview": f"{grok_key[:8]}...{grok_key[-4:]}" if grok_key and len(grok_key) > 12 else "Invalid",
        "base_url": GROK_BASE_URL,
        "primary_model": PRIMARY_CHAT_MODEL,
        "client_initialized": bool(self.grok_client),
        "is_healthy": self.is_healthy,
        "last_error": self.last_error
    }


# ==================== Production Singleton ====================

try:
    ai_service = AIService()
    logger.info(f"✅ AI Service instance created with model: {ai_service.current_chat_model}")
except Exception as e:
    logger.critical(f"❌ CRITICAL: Unexpected exception creating AI service instance: {e}")
    try:
        ai_service = AIService()
    except Exception:
        ai_service = None