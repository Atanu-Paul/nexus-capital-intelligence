"""Loads PDFs/txt, chunks documents."""

import re
from pathlib import Path

from pypdf import PdfReader

COMPANY_NAMES = [
    "NovaTech Systems",
    "Quantix Software",
    "Orbital Cloud Inc.",
    "Vertex Semiconductors",
    "Lumen Data Corp",
    "Zenith National Bank",
    "Meridian Trust Bank",
    "Ashford Financial Group",
    "Highline Capital Bank",
    "Sterling Reserve Bank",
    "Cascade Biopharma",
    "Novagen Therapeutics",
    "Harborview Pharmaceuticals",
    "Elysium Life Sciences",
    "Pinnacle Health Corp",
]


def _company_slug(name: str) -> str:
    """Convert a company name to the filename slug used in synthetic_data/."""
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s]+", "_", slug.strip())
    return slug


SLUG_TO_COMPANY = {_company_slug(name): name for name in COMPANY_NAMES}


def load_pdf_text(filepath: str) -> str:
    """Extract all text from a PDF file.

    Args:
        filepath: Path to the PDF file.

    Returns:
        Combined text from all pages.
    """
    reader = PdfReader(filepath)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def load_txt_text(filepath: str) -> str:
    """Read all text from a plain text file.

    Args:
        filepath: Path to the text file.

    Returns:
        File contents as a string.
    """
    return Path(filepath).read_text(encoding="utf-8").strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-based chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += chunk_size - overlap

    return chunks


def _filename_to_slug(filename: str) -> str:
    """Extract the company slug from a synthetic data filename."""
    if filename.endswith("_q3report.pdf"):
        return filename[: -len("_q3report.pdf")]
    if filename.endswith("_news.txt"):
        return filename[: -len("_news.txt")]
    return Path(filename).stem


def infer_company_name(filename: str) -> str:
    """Infer the company display name from a synthetic data filename."""
    slug = _filename_to_slug(filename)
    return SLUG_TO_COMPANY.get(slug, slug.replace("_", " ").title())


def load_all_documents(base_dir: str = "synthetic_data") -> list[dict]:
    """Load and chunk all PDF and text documents under the synthetic data directory.

    Args:
        base_dir: Root directory containing sector subfolders.

    Returns:
        List of chunk dictionaries with text and metadata.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Synthetic data directory not found: {base_dir}")

    documents: list[dict] = []

    for sector_dir in sorted(base_path.iterdir()):
        if not sector_dir.is_dir() or sector_dir.name.startswith("."):
            continue

        sector = sector_dir.name

        for filepath in sorted(sector_dir.iterdir()):
            if filepath.name.startswith("."):
                continue

            suffix = filepath.suffix.lower()
            if suffix == ".pdf":
                text = load_pdf_text(str(filepath))
                doc_type = "report"
            elif suffix == ".txt":
                text = load_txt_text(str(filepath))
                doc_type = "news"
            else:
                continue

            company = infer_company_name(filepath.name)
            source_file = filepath.name

            for chunk in chunk_text(text):
                documents.append(
                    {
                        "chunk_text": chunk,
                        "source_file": source_file,
                        "company": company,
                        "sector": sector,
                        "doc_type": doc_type,
                    }
                )

    return documents
