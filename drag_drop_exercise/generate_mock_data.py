from typing import Dict, Any
from drag_drop_exercise.mock_data import BoundingBox, combine_bounding_boxes, MockPdf, MockImage


bounding_boxes_dict = {
    "geometry": {"text": "Geometry", "x": 46, "y": 42, "width": 208, "height": 42},
    "estimation1": {"text": "Estimation", "x": 268, "y": 41, "width": 218, "height": 34},
    "image": {"text": "Image", "x": 567, "y": 43, "width": 122, "height": 41},
    "denoising": {"text": "Denoising", "x": 706, "y": 41, "width": 202, "height": 43},
    "object": {"text": "Object", "x": 1032, "y": 40, "width": 135, "height": 44},
    "segmentation": {"text": "Segmentation", "x": 1180, "y": 41, "width": 287, "height": 43},
    "depth": {"text": "Depth", "x": 1547, "y": 40, "width": 123, "height": 44},
    "estimation2": {"text": "Estimation", "x": 1687, "y": 41, "width": 218, "height": 34},
}

bounding_boxes_dict_2 = {
    "summing": {"text": "Summing", "x": 557, "y": 90, "width": 164, "height": 48},
    "junction": {"text": "junction", "x": 566, "y": 142, "width": 142, "height": 39},
    "input": {"text": "Input", "x": 41, "y": 289, "width": 98, "height": 63},
    "output": {"text": "Output", "x": 1040, "y": 279, "width": 115, "height": 50},
    "signals": {"text": "signals", "x": 23, "y": 337, "width": 118, "height": 39},
    "activation": {"text": "Activation", "x": 775, "y": 393, "width": 179, "height": 31},
    "function": {"text": "function", "x": 794, "y": 441, "width": 141, "height": 32},
}

# IMAGE 1
geometry_estimation = combine_bounding_boxes(bounding_boxes_dict["geometry"], bounding_boxes_dict["estimation1"])
image_denoising = combine_bounding_boxes(bounding_boxes_dict["image"], bounding_boxes_dict["denoising"])
object_segmentation = combine_bounding_boxes(bounding_boxes_dict["object"], bounding_boxes_dict["segmentation"])
depth_estimation = combine_bounding_boxes(bounding_boxes_dict["depth"], bounding_boxes_dict["estimation2"])

# IMAGE 2
summing_junction = combine_bounding_boxes(bounding_boxes_dict_2["summing"], bounding_boxes_dict_2["junction"])
input_signals = combine_bounding_boxes(bounding_boxes_dict_2["input"], bounding_boxes_dict_2["signals"])
output = bounding_boxes_dict_2["output"]
activation_function = combine_bounding_boxes(bounding_boxes_dict_2["activation"], bounding_boxes_dict_2["function"])


pdf_name = "ML4Lecture02image.pdf"

mockimage1 = MockImage(
    image_name="image1",
    image_path="images/image1.png",
    text_bounding_boxes=[
        geometry_estimation,
        image_denoising,
        object_segmentation,
        depth_estimation
    ],
)

mockimage2 = MockImage(
    image_name="neural net",
    image_path="images/image.png",
    text_bounding_boxes=[
        summing_junction,
        input_signals,
        output,
        activation_function
    ],
)

mock_pdf = MockPdf(
    pdf_name=pdf_name,
    images=[mockimage1, mockimage2]
)

model_json = mock_pdf.model_dump_json()
# write to file
with open('mock_pdf.json', 'w') as f:
    f.write(model_json)