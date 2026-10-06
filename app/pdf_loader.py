from pathlib import Path

from app.document_loader import load_pdf

def load_pdf_text(pdf_path: str) -> str:
    return load_pdf(Path(pdf_path))
