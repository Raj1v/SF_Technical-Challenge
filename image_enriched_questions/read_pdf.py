import fitz  # PyMuPDF
from pydantic import BaseModel
from typing import List, Dict
import os


class PdfData(BaseModel):
    text: str = ""
    images: List[str] = []
    image_descriptions: List[str] = []
    relevant_images: Dict[str, str] | None = None

def extract_text_and_images_from_pages(
    pdf_path, start_page, end_page, image_output_folder
) -> PdfData:
    pdfData = PdfData()
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(start_page - 1, end_page):
        page = doc.load_page(page_num)
        pages.append(page)

    for page in pages:
        text = page.get_text()

        pdfData.text += f"\n\nPAGE {page.number + 1}:\n\n"
        pdfData.text += text + "\n"

        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = f"page_{page.number + 1}_image_{img_index}.{image_ext}"
            image_path = os.path.join(image_output_folder, image_path)

            with open(
                image_path, "wb"
            ) as img_file:
                img_file.write(image_bytes)

            pdfData.images.append(image_path)

    return pdfData


