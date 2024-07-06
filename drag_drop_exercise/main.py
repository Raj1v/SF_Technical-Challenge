# BIG Anti-pattern: appending to sys.path to import modules from other directories
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dotenv
from image_to_labels_pipeline import describe_diagram, get_text_labels
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from utils import image_to_human_message
from typing import List

dotenv.load_dotenv(".env")
dotenv.load_dotenv(".secrets")

@chain
def image_to_drag_and_drop_exercise(image: HumanMessage):
    image_path = image.additional_kwargs["image_path"]
    describe_diagram.invoke(image)
    return get_text_labels.invoke(image_path)


image_paths = ["images/image1.png", "images/image.png"]


@chain
def run_pipeline(images: List[str]):
    for image_path in images:
        message = image_to_human_message(text="Describe this diagram",
                                         image_path=image_path)
        image_to_drag_and_drop_exercise.invoke(message)

run_pipeline.invoke(image_paths, config={"run_name": "Create drag and drop exercises"})