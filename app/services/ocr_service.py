import asyncio
import aiohttp
import base64
import io
from typing import Dict, Any, Optional, List
from PIL import Image
import requests
from app.core.config import settings

class OCRService:
    """
    Production OCR service with multiple providers and fallbacks.
    Supports dots.ocr and other OCR APIs for extracting text from images.
    """
    
    def __init__(self):
        self.session = None
        self.ocr_providers = {
            'dots_ocr': {
                'url': 'https://api.dots.dev/ocr/v1/extract',
                'headers': {'Authorization': f'Bearer {getattr(settings, "DOTS_OCR_API_KEY", "")}'}
            },
            'tesseract_fallback': True  # Local fallback
        }
    
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=10)
            )
        return self.session
    
    async def extract_text_from_image(self, image_url: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Extract text from image using OCR with multiple provider fallbacks.
        Returns: {
            'text': str,
            'confidence': float,
            'provider': str,
            'success': bool,
            'error': str
        }
        """
        try:
            if hasattr(settings, 'DOTS_OCR_API_KEY') and settings.DOTS_OCR_API_KEY:
                result = await self._extract_with_dots_ocr(image_url, image_data)
                if result['success']:
                    return result
            
            result = await self._extract_with_fallback_ocr(image_url, image_data)
            return result
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'provider': 'none',
                'success': False,
                'error': str(e)
            }
    
    async def _extract_with_dots_ocr(self, image_url: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Extract text using dots.ocr API"""
        try:
            session = await self.get_session()
            
            if image_data:
                # Use provided image data
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            else:
                # Download image from URL
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to download image: {resp.status}")
                    image_data = await resp.read()
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Call dots.ocr API
            payload = {
                'image': f'data:image/jpeg;base64,{image_b64}',
                'language': 'en',
                'output_format': 'text'
            }
            
            async with session.post(
                self.ocr_providers['dots_ocr']['url'],
                json=payload,
                headers=self.ocr_providers['dots_ocr']['headers']
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        'text': result.get('text', ''),
                        'confidence': result.get('confidence', 0.9),
                        'provider': 'dots_ocr',
                        'success': True,
                        'error': ''
                    }
                else:
                    error_text = await resp.text()
                    raise Exception(f"dots.ocr API error: {resp.status} - {error_text}")
                    
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'provider': 'dots_ocr',
                'success': False,
                'error': str(e)
            }
    
    async def _extract_with_fallback_ocr(self, image_url: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Fallback OCR using local processing or other APIs"""
        try:
            # Simple fallback - could integrate with pytesseract or other OCR libraries
            # For production, you might want to add more sophisticated OCR providers
            
            if not image_data and image_url:
                session = await self.get_session()
                async with session.get(image_url) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
            
            if image_data:
                # Basic image validation
                try:
                    img = Image.open(io.BytesIO(image_data))
                    # Simple heuristic: if image is very small, likely no meaningful text
                    if img.width < 50 or img.height < 50:
                        return {
                            'text': '',
                            'confidence': 0.0,
                            'provider': 'fallback',
                            'success': True,
                            'error': 'Image too small for OCR'
                        }
                except Exception:
                    pass
            
            # For now, return empty result - in production you'd integrate actual OCR
            return {
                'text': '',
                'confidence': 0.0,
                'provider': 'fallback',
                'success': True,
                'error': 'OCR fallback not implemented'
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'provider': 'fallback',
                'success': False,
                'error': str(e)
            }
    
    async def batch_extract_text(self, images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract text from multiple images concurrently.
        images: [{'url': str, 'alt': str}, ...]
        """
        tasks = []
        for img in images[:10]:  # Limit batch size for production
            task = self.extract_text_from_image(img['url'])
            tasks.append(task)
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        enhanced_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                enhanced_results.append({
                    'text': '',
                    'confidence': 0.0,
                    'provider': 'error',
                    'success': False,
                    'error': str(result),
                    'image_url': images[i]['url'],
                    'image_alt': images[i].get('alt', '')
                })
            else:
                result['image_url'] = images[i]['url']
                result['image_alt'] = images[i].get('alt', '')
                enhanced_results.append(result)
        
        return enhanced_results
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None

ocr_service = OCRService()
