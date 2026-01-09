# image_service.py
from app.services.ai_service import ai_service

class ImageService:
    async def generate_image(self, prompt: str) -> bytes:
        img_data = await ai_service.generate_image(prompt)
        return img_data

image_service = ImageService()
