# file_service.py
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

class FileService:
    async def generate_pdf(self, text: str, title: str = "AI Report") -> bytes:
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        pdf.setTitle(title)
        pdf.drawString(100, 800, title)
        text_obj = pdf.beginText(50, 780)
        for line in text.split("\n"):
            text_obj.textLine(line)
        pdf.drawText(text_obj)
        pdf.save()
        buffer.seek(0)
        return buffer.getvalue()

file_service = FileService()
