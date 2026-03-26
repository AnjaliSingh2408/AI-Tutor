from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ExtractedDiagram:
    source_pdf: str
    page_num: int
    image_name: str
    image_path: Path
    width: int
    height: int


def extract_pdf_diagrams(
    *,
    books_dir: str | Path,
    out_dir: str | Path,
    min_width: int = 180,
    min_height: int = 120,
) -> list[ExtractedDiagram]:
    """
    Extract embedded images from textbook PDFs for multimodal indexing.

    This catches diagrams that are stored as image objects in PDF pages.
    """
    books_dir = Path(books_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(books_dir.rglob("*.pdf"))
    extracted: list[ExtractedDiagram] = []

    for pdf in pdfs:
        try:
            doc = fitz.open(pdf)
        except Exception:
            continue

        pdf_stem = pdf.stem
        try:
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                images = page.get_images(full=True)
                for img_i, img in enumerate(images):
                    xref = img[0]
                    try:
                        info = doc.extract_image(xref)
                    except Exception:
                        continue
                    blob = info.get("image")
                    ext = info.get("ext", "png")
                    width = int(info.get("width") or 0)
                    height = int(info.get("height") or 0)
                    if not blob or width < min_width or height < min_height:
                        continue
                    out_name = f"{pdf_stem}_p{page_idx + 1:03d}_{img_i:02d}.{ext}"
                    out_path = out_dir / out_name
                    out_path.write_bytes(blob)
                    extracted.append(
                        ExtractedDiagram(
                            source_pdf=pdf.name,
                            page_num=page_idx + 1,
                            image_name=out_name,
                            image_path=out_path,
                            width=width,
                            height=height,
                        )
                    )
        finally:
            doc.close()

    return extracted
