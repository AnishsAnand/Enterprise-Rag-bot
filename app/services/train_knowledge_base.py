import os
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from fpdf import FPDF



CHROMA_PATH = "chroma_db"
PDF_OUTPUT_DIR = "pdfs"
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def save_images_to_pdf(images, pdf_filename):
    """Save scraped images into a PDF file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for img_url in images:
        try:
            response = requests.get(img_url, stream=True, timeout=10)
            if response.status_code == 200:
                img_file = os.path.join(PDF_OUTPUT_DIR, os.path.basename(img_url.split("?")[0]))
                with open(img_file, "wb") as f:
                    f.write(response.content)

                pdf.add_page()
                pdf.image(img_file, x=10, y=20, w=180)
        except Exception as e:
            print(f"[WARN] Failed to save image {img_url}: {e}")

    pdf.output(os.path.join(PDF_OUTPUT_DIR, pdf_filename))
    print(f"[INFO] PDF saved: {pdf_filename}")



def scrape_and_train(url: str):
    """Scrape webpage text + images, store embeddings in Chroma, and save images to PDF."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    images = [img["src"] for img in soup.find_all("img") if img.get("src")]
    images = [img if img.startswith("http") else requests.compat.urljoin(url, img) for img in images]
    pdf_filename = f"{url.split('//')[-1].replace('/', '_')}.pdf"
    save_images_to_pdf(images, pdf_filename)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PATH)
    vectorstore.persist()

    print(f"[INFO] Scraped and trained from {url}. Chroma DB updated.")



def add_pdf_to_vectorstore(pdf_path: str):
    """Extract text from PDF and add to Chroma DB."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PATH)
    vectorstore.persist()
    print(f"[INFO] Added {pdf_path} to Chroma DB.")
