from pydantic import BaseModel
from typing import List
from typing_extensions import TypedDict
import json


class BoundingBox(TypedDict):
    text: str
    x: int
    y: int
    width: int
    height: int


def combine_bounding_boxes(box1: BoundingBox, box2: BoundingBox) -> BoundingBox:
    combined_text = box1["text"] + " " + box2["text"]
    x_min = min(box1["x"], box2["x"])
    y_min = min(box1["y"], box2["y"])
    x_max = max(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y_max = max(box1["y"] + box1["height"], box2["y"] + box2["height"])
    combined_width = x_max - x_min
    combined_height = y_max - y_min

    return BoundingBox(
        text=combined_text,
        x=x_min,
        y=y_min,
        width=combined_width,
        height=combined_height,
    )


class MockPdf(BaseModel):
    pdf_name: str
    images: List["MockImage"]


class MockImage(BaseModel):
    image_name: str
    image_path: str
    text_bounding_boxes: List[BoundingBox]


path = "drag_drop_exercise/mock_pdf.json"
with open(path, "r") as f:
    data = json.load(f)
mock_pdf_1 = MockPdf(**data)

mock_pdfs = [mock_pdf_1]


def get_mock_pdf(pdf_name: str) -> MockPdf:
    for pdf in mock_pdfs:
        if pdf.pdf_name == pdf_name:
            return pdf
    raise ValueError(f"Mock PDF not found for {pdf_name}")


def get_mock_image(image_path: str) -> MockImage:
    image_path = image_path.replace("drag_drop_exercise/", "")
    for pdf in mock_pdfs:
        for image in pdf.images:
            if image.image_path == image_path:
                return image
    raise ValueError(f"Mock Image not found for {image_path}")