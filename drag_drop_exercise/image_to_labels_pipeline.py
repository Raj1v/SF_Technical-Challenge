from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
import pytesseract
from pytesseract import Output
import cv2
from utils import image_to_human_message
from drag_drop_exercise.mock_data import MockImage, get_mock_image, BoundingBox
from typing import List, Dict, Union, Tuple

@chain
def describe_diagram(message: HumanMessage):
    model = ChatOpenAI(model="gpt-4o")

    response = model.invoke([message])
    return response

@chain
def get_text_labels(image_path: str):
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Use Tesseract to detect text
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(binary, output_type=Output.DICT, config=custom_config)

    # Initialize the list of bounding boxes and text
    text_boxes = []

    # Loop over each of the text localizations
    for i in range(len(details['text'])):
        if int(details['conf'][i]) > 60:  # Filter out weak confidence text detections
            x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
            text = details['text'][i]
            text_boxes.append({
                'text': text,
                'bounding_box': (x, y, w, h)
            })
            # Draw the bounding box on the image (optional, for visualization)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    mock_bounding_boxes = get_mock_image(image_path).text_bounding_boxes
    output_image = 'output.png'
    draw_bounding_boxes(image_path, mock_bounding_boxes, output_image)
    message = image_to_human_message(text="Text labels detected", image_path=output_image)
    message.additional_kwargs['bounding_boxes-real'] = text_boxes
    message.additional_kwargs['bounding_boxes-mock'] = mock_bounding_boxes

    return message


def draw_bounding_boxes(image_path: str,
                        bounding_boxes: List[BoundingBox],
                        output_path: str = 'output.png'):
    image = cv2.imread(image_path)
    for box in bounding_boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite('output.png', image)