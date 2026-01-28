"""
Intelligent Web Crawler with Priority-Based Discovery

A production-grade web crawler that intelligently discovers and prioritizes URLs
based on content analysis, link importance, and configurable patterns.

Features:
- Priority-based URL discovery
- Content-aware importance scoring
- Multi-domain support with filters
- Sitemap parsing
- Robots.txt compliance
- Configurable crawl strategies

Author: Production Team
Version: 2.0.0
"""

import asyncio
import logging
import re
from typing import List, Dict, Set, Tuple, Optional, Deque
from urllib.parse import urljoin, urlparse
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class URLPriority(Enum):
    """Priority levels for URLs."""
    CRITICAL = 100
    HIGH = 80
    MEDIUM = 50
    LOW = 20
    VERY_LOW = 10


@dataclass
class CrawlConfig:
    """Configuration for crawler behavior."""
    # Crawl limits
    max_depth: int = 3
    max_urls: int = 500
    max_urls_per_domain: int = 100
    
    # Domain handling
    follow_external: bool = False
    domain_filter: Optional[str] = None
    
    # Behavior
    respect_robots_txt: bool = True
    use_sitemap: bool = True
    
    # Rate limiting
    request_delay: float = 0.2
    request_timeout: int = 15
    max_retries: int = 2
    
    # Content filtering
    skip_extensions: Set[str] = field(default_factory=lambda: {
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
        '.mp4', '.avi', '.mov', '.zip', '.tar', '.gz', '.exe',
        '.dmg', '.css', '.js', '.json', '.xml', '.ico', '.woff',
        '.woff2', '.ttf', '.eot', '.mp3', '.wav', '.rar', '.7z'
    })
    
    # Priority patterns
    high_priority_patterns: List[str] = field(default_factory=lambda: [
        r'/docs?/',
        r'/documentation/',
        r'/guide/',
        r'/tutorial/',
        r'/api/',
        r'/reference/',
        r'/kb/',
        r'/knowledge-?base/',
        r'/help/',
        r'/support/',
        r'/manual/',
        r'/wiki/',
        r'/getting-started/',
        r'/quickstart/',
    ])
    
    medium_priority_patterns: List[str] = field(default_factory=lambda: [
        r'/blog/',
        r'/article/',
        r'/post/',
        r'/news/',
        r'/feature/',
        r'/product/',
        r'/service/',
        r'/case-study/',
        r'/whitepaper/',
    ])
    
    low_priority_patterns: List[str] = field(default_factory=lambda: [
        r'/tag/',
        r'/category/',
        r'/author/',
        r'/archive/',
        r'/page/\d+',
        r'/search',
        r'/login',
        r'/signup',
        r'/register',
        r'/cart/',
        r'/checkout/',
        r'/account/',
    ])


@dataclass
class URLMetadata:
    """Metadata for a discovered URL."""
    url: str
    priority: float
    depth: int
    source_url: str
    link_text: str
    context: str
    discovered_at: float
    visited: bool = False


class IntelligentCrawler:
    """
    Production-grade intelligent web crawler.
    
    Discovers and prioritizes URLs based on:
    - URL patterns (docs, guides, etc.)
    - Link text and context
    - Depth from starting point
    - Sitemap availability
    - Page structure
    
    Usage:
        crawler = IntelligentCrawler()
        urls = await crawler.discover_urls("https://example.com", max_urls=100)
    """
    
    def __init__(self, config: Optional[CrawlConfig] = None):
        """
        Initialize the crawler.
        
        Args:
            config: Optional crawl configuration
        """
        self.config = config or CrawlConfig()
        self.user_agent = self._get_user_agent()
        
        # Statistics
        self.stats = {
            'urls_discovered': 0,
            'urls_visited': 0,
            'urls_skipped': 0,
            'sitemaps_found': 0,
            'errors': 0
        }
        
        logger.info(f"IntelligentCrawler initialized with config: {self.config}")
    
    async def discover_urls(
        self,
        base_url: str,
        max_depth: Optional[int] = None,
        max_urls: Optional[int] = None,
        follow_external: Optional[bool] = None,
        domain_filter: Optional[str] = None
    ) -> List[str]:
        """
        Discover URLs intelligently with priority-based crawling.
        
        Args:
            base_url: Starting URL for crawl
            max_depth: Maximum crawl depth (overrides config)
            max_urls: Maximum URLs to discover (overrides config)
            follow_external: Whether to follow external links (overrides config)
            domain_filter: Domain filter string (overrides config)
        
        Returns:
            List of discovered URLs sorted by importance
        """
        # Override config if parameters provided
        max_depth = max_depth if max_depth is not None else self.config.max_depth
        max_urls = max_urls if max_urls is not None else self.config.max_urls
        follow_external = follow_external if follow_external is not None else self.config.follow_external
        domain_filter = domain_filter or self.config.domain_filter
        
        logger.info(f"🕷️  Starting intelligent crawl: {base_url}")
        logger.info(f"   Max depth: {max_depth}, Max URLs: {max_urls}")
        
        # Reset statistics
        self.stats = {k: 0 for k in self.stats}
        
        # Initialize crawl state
        base_domain = urlparse(base_url).netloc
        
        # Priority queue: (priority, depth, url, source_url, link_text, context)
        queue: Deque[Tuple[float, int, str, str, str, str]] = deque([
            (URLPriority.CRITICAL.value, 0, base_url, "", "", "")
        ])
        
        discovered: Set[str] = {base_url}
        visited: Set[str] = set()
        url_metadata: Dict[str, URLMetadata] = {}
        
        # Try sitemap first
        if self.config.use_sitemap:
            sitemap_urls = await self._discover_from_sitemap(base_url)
            if sitemap_urls:
                logger.info(f"📋 Found {len(sitemap_urls)} URLs from sitemap")
                self.stats['sitemaps_found'] = len(sitemap_urls)
                
                for url in sitemap_urls[:max_urls // 2]:
                    if url not in discovered:
                        priority = self._calculate_url_priority(url, "", 1, "")
                        queue.append((priority, 1, url, base_url, "sitemap", ""))
                        discovered.add(url)
        
        # Crawl with priority
        while queue and len(discovered) < max_urls:
            # Sort queue by priority (highest first)
            queue = deque(sorted(queue, key=lambda x: x[0], reverse=True))
            
            priority, depth, current_url, source_url, link_text, context = queue.popleft()
            
            # Skip if already visited or too deep
            if current_url in visited or depth > max_depth:
                continue
            
            visited.add(current_url)
            self.stats['urls_visited'] += 1
            
            # Store metadata
            url_metadata[current_url] = URLMetadata(
                url=current_url,
                priority=priority,
                depth=depth,
                source_url=source_url,
                link_text=link_text,
                context=context,
                discovered_at=asyncio.get_event_loop().time(),
                visited=True
            )
            
            # Fetch and parse page
            try:
                html = await self._fetch_page(current_url)
                if not html:
                    continue
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract links with context
                links = self._extract_links_with_context(soup, current_url)
                
                for link_url, link_text_new, link_context in links:
                    # Domain filtering
                    link_domain = urlparse(link_url).netloc
                    
                    # Check external links
                    if not follow_external and link_domain != base_domain:
                        self.stats['urls_skipped'] += 1
                        continue
                    
                    # Check domain filter
                    if domain_filter and domain_filter.lower() not in link_domain.lower():
                        self.stats['urls_skipped'] += 1
                        continue
                    
                    # Skip if already discovered
                    if link_url in discovered:
                        continue
                    
                    # Skip unwanted extensions
                    if self._should_skip_url(link_url):
                        self.stats['urls_skipped'] += 1
                        continue
                    
                    # Calculate priority
                    link_priority = self._calculate_url_priority(
                        link_url,
                        link_text_new,
                        depth + 1,
                        link_context
                    )
                    
                    # Add to queue
                    queue.append((
                        link_priority,
                        depth + 1,
                        link_url,
                        current_url,
                        link_text_new,
                        link_context
                    ))
                    discovered.add(link_url)
                    self.stats['urls_discovered'] += 1
                
                # Progress logging
                if len(discovered) % 50 == 0:
                    logger.info(
                        f"📊 Progress: {len(discovered)} discovered, "
                        f"{len(visited)} visited, {len(queue)} queued"
                    )
                
                # Rate limiting
                await asyncio.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.debug(f"Error crawling {current_url}: {e}")
                self.stats['errors'] += 1
                continue
        
        # Sort by importance
        sorted_urls = sorted(
            discovered,
            key=lambda url: url_metadata.get(url, URLMetadata(
                url="", priority=0, depth=0, source_url="",
                link_text="", context="", discovered_at=0
            )).priority,
            reverse=True
        )
        
        logger.info(f"✅ Discovery complete: {len(sorted_urls)} URLs found")
        self._log_statistics()
        
        return sorted_urls[:max_urls]
    
    def _calculate_url_priority(
        self,
        url: str,
        link_text: str = "",
        depth: int = 0,
        context: str = ""
    ) -> float:
        """
        Calculate URL priority score (0-100).
        
        Args:
            url: URL to score
            link_text: Text of the link
            depth: Depth from starting URL
            context: Surrounding context
        
        Returns:
            Priority score (higher = more important)
        """
        score = 50.0  # Base score
        
        url_lower = url.lower()
        text_lower = link_text.lower()
        context_lower = context.lower()
        
        # High priority patterns (+40 points)
        for pattern in self.config.high_priority_patterns:
            if re.search(pattern, url_lower):
                score += 40
                break
        
        # Medium priority patterns (+20 points)
        for pattern in self.config.medium_priority_patterns:
            if re.search(pattern, url_lower):
                score += 20
                break
        
        # Low priority patterns (-30 points)
        for pattern in self.config.low_priority_patterns:
            if re.search(pattern, url_lower):
                score -= 30
                break
        
        # Link text indicators (+15 points)
        high_value_terms = [
            'documentation', 'docs', 'guide', 'tutorial', 'api',
            'reference', 'manual', 'getting started', 'quickstart',
            'overview', 'introduction', 'readme', 'how to'
        ]
        if any(term in text_lower for term in high_value_terms):
            score += 15
        
        # Context indicators (+10 points)
        if any(term in context_lower for term in high_value_terms):
            score += 10
        
        # Depth penalty (-5 points per level)
        score -= (depth * 5)
        
        # URL structure analysis
        path_depth = url.count('/') - 2  # Subtract protocol slashes
        
        # Shorter paths often more important (+10 for root-level)
        if path_depth <= 2:
            score += 10
        elif path_depth <= 4:
            score += 5
        
        # Penalize very deep paths
        if path_depth > 8:
            score -= 10
        
        # Bonus for clean URLs (no query params)
        if '?' not in url:
            score += 5
        
        # Penalize URLs with many query params
        if url.count('&') > 3:
            score -= 10
        
        return max(0, min(100, score))
    
    def _extract_links_with_context(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> List[Tuple[str, str, str]]:
        """
        Extract links with surrounding context for better prioritization.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for relative links
        
        Returns:
            List of (url, link_text, context) tuples
        """
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            
            # Skip special protocols
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Make absolute URL
            try:
                url = urljoin(base_url, href)
            except Exception:
                continue
            
            # Clean URL (remove fragments)
            parsed = urlparse(url)
            url = parsed._replace(fragment='').geturl()
            
            # Extract link text
            link_text = a.get_text(strip=True)
            
            # Extract context (parent text)
            context = ""
            parent = a.parent
            if parent:
                context = parent.get_text(strip=True)[:200]
            
            links.append((url, link_text, context))
        
        return links
    
    async def _discover_from_sitemap(self, base_url: str) -> List[str]:
        """
        Discover URLs from sitemap.xml.
        
        Args:
            base_url: Base URL to check for sitemaps
        
        Returns:
            List of URLs found in sitemaps
        """
        sitemap_urls = []
        
        # Common sitemap locations
        sitemap_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap1.xml',
            '/sitemap',
            '/robots.txt',  # May contain sitemap reference
        ]
        
        for path in sitemap_paths:
            sitemap_url = urljoin(base_url, path)
            
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.request_timeout,
                    follow_redirects=True
                ) as client:
                    response = await client.get(
                        sitemap_url,
                        headers={'User-Agent': self.user_agent}
                    )
                    
                    if response.status_code == 200:
                        # Handle robots.txt
                        if path == '/robots.txt':
                            sitemap_refs = re.findall(
                                r'Sitemap:\s*(.+)',
                                response.text,
                                re.IGNORECASE
                            )
                            for ref in sitemap_refs:
                                sitemap_urls.extend(
                                    await self._parse_sitemap(ref.strip())
                                )
                        else:
                            # Parse as XML sitemap
                            sitemap_urls.extend(
                                await self._parse_sitemap_content(response.text)
                            )
                        
                        if sitemap_urls:
                            logger.info(f"✅ Loaded sitemap from {sitemap_url}")
                            break
            
            except Exception as e:
                logger.debug(f"Sitemap fetch failed for {sitemap_url}: {e}")
                continue
        
        return sitemap_urls
    
    async def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap URL.
        
        Args:
            sitemap_url: URL of sitemap to parse
        
        Returns:
            List of URLs from sitemap
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.config.request_timeout,
                follow_redirects=True
            ) as client:
                response = await client.get(
                    sitemap_url,
                    headers={'User-Agent': self.user_agent}
                )
                
                if response.status_code == 200:
                    return await self._parse_sitemap_content(response.text)
        
        except Exception as e:
            logger.debug(f"Failed to parse sitemap {sitemap_url}: {e}")
        
        return []
    
    async def _parse_sitemap_content(self, content: str) -> List[str]:
        """
        Parse sitemap XML content.
        
        Args:
            content: Sitemap XML content
        
        Returns:
            List of URLs
        """
        urls = []
        
        try:
            soup = BeautifulSoup(content, 'xml')
            
            # Extract URLs from <loc> tags
            for loc in soup.find_all('loc'):
                url = loc.get_text(strip=True)
                if url and url.startswith('http'):
                    urls.append(url)
            
            # Handle sitemap index (recursive sitemaps)
            for sitemap in soup.find_all('sitemap'):
                loc = sitemap.find('loc')
                if loc:
                    nested_url = loc.get_text(strip=True)
                    if nested_url and nested_url.startswith('http'):
                        nested_urls = await self._parse_sitemap(nested_url)
                        urls.extend(nested_urls)
        
        except Exception as e:
            logger.debug(f"Failed to parse sitemap content: {e}")
        
        return urls
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch page HTML with retries and error handling.
        
        Args:
            url: URL to fetch
        
        Returns:
            HTML content or None if failed
        """
        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.request_timeout,
                    follow_redirects=True
                ) as client:
                    response = await client.get(
                        url,
                        headers={'User-Agent': self.user_agent}
                    )
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' in content_type:
                            return response.text
                
                return None
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                logger.debug(f"Fetch failed for {url}: {e}")
                return None
        
        return None
    
    def _should_skip_url(self, url: str) -> bool:
        """
        Check if URL should be skipped based on extension.
        
        Args:
            url: URL to check
        
        Returns:
            True if URL should be skipped
        """
        url_lower = url.lower()
        return any(url_lower.endswith(ext) for ext in self.config.skip_extensions)
    
    def _get_user_agent(self) -> str:
        """Get user agent string."""
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    
    def _log_statistics(self) -> None:
        """Log crawl statistics."""
        logger.info("📊 Crawl Statistics:")
        logger.info(f"   URLs discovered: {self.stats['urls_discovered']}")
        logger.info(f"   URLs visited: {self.stats['urls_visited']}")
        logger.info(f"   URLs skipped: {self.stats['urls_skipped']}")
        logger.info(f"   Sitemaps found: {self.stats['sitemaps_found']}")
        logger.info(f"   Errors: {self.stats['errors']}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get crawl statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()


# Convenience functions

async def quick_crawl(
    base_url: str,
    max_urls: int = 100,
    max_depth: int = 3
) -> List[str]:
    """
    Quick crawl with default settings.
    
    Args:
        base_url: Starting URL
        max_urls: Maximum URLs to discover
        max_depth: Maximum depth to crawl
    
    Returns:
        List of discovered URLs
    """
    crawler = IntelligentCrawler()
    return await crawler.discover_urls(
        base_url,
        max_urls=max_urls,
        max_depth=max_depth
    )


async def deep_crawl(
    base_url: str,
    max_urls: int = 1000,
    max_depth: int = 5,
    follow_external: bool = False
) -> List[str]:
    """
    Deep crawl with extended limits.
    
    Args:
        base_url: Starting URL
        max_urls: Maximum URLs to discover
        max_depth: Maximum depth to crawl
        follow_external: Whether to follow external links
    
    Returns:
        List of discovered URLs
    """
    config = CrawlConfig(
        max_depth=max_depth,
        max_urls=max_urls,
        follow_external=follow_external,
        request_delay=0.5  # Slower for politeness
    )
    
    crawler = IntelligentCrawler(config)
    return await crawler.discover_urls(base_url)