# BIG Anti-pattern: appending to sys.path to import modules from other directories
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from typing import List
from typing_extensions import TypedDict
from utils import image_to_human_message
from read_pdf import extract_text_and_images_from_pages, PdfData
import dotenv

dotenv.load_dotenv(".env")
dotenv.load_dotenv(".secrets")


class PipelineInput(TypedDict):
    pdf_path: str
    start_page: int
    end_page: int


@chain
def read_pdf(input: PipelineInput) -> PdfData:
    pdf_path = input["pdf_path"]
    start_page = input["start_page"]
    end_page = input["end_page"]
    img_output_folder = "image_enriched_questions/images"
    pdfData = extract_text_and_images_from_pages(
        pdf_path, start_page, end_page, img_output_folder
    )
    return pdfData


@chain
def clean_up_text(pdfData: PdfData) -> PdfData:
    # Make call to
    model = ChatOpenAI(model="gpt-4o")
    text = pdfData.text
    prompt = (
        "Below is a raw extraction from a PDF, clean up the text into properly structured sentences.\n\n"
        "Text: {text}"
    )
    prompt_template = ChatPromptTemplate.from_messages([("system", prompt)])
    chain = prompt_template | model | StrOutputParser()
    cleaned_text = chain.invoke({"text": text})
    pdfData.text = cleaned_text
    return pdfData


@chain
def describe_images(pdfData: PdfData) -> PdfData:
    model = ChatOpenAI(model="gpt-4o")
    chain = model | StrOutputParser()

    for image_path in pdfData.images:
        message = image_to_human_message(
            text="Describe this image", image_path=image_path
        )
        messages = [message]
        response = chain.invoke(messages)
        pdfData.image_descriptions.append(response)

    return pdfData


@chain
def mock_relevant_image_selection(pdfData: PdfData) -> PdfData:
    # Mock relevant image selection
    # This should be some LLM model that selects the most relevant images
    relevant_images = {
        pdfData.images[-1]: pdfData.image_descriptions[-1],
        pdfData.images[-2]: pdfData.image_descriptions[-2],
    }
    pdfData.relevant_images = relevant_images
    return pdfData


@chain
def generate_questions(pdfData: PdfData):
    model = ChatOpenAI(
        model="gpt-4o", model_kwargs={"response_format": {"type": "json_object"}}
    )
    text = pdfData.text
    image_descriptions = ""
    for i, description in enumerate(pdfData.relevant_images.values()):
        image_descriptions += f"Image Description {i}: {description}\n\n"

    prompt1 = SystemMessage(
        "Below is the text from a PDF and the description of images that you previously selected as relevant\n\n"
        f"Text: {text} \n\n"
    )
    prompt2 = SystemMessage(f"IMAGE DESCRIPTIONS:\n\n {image_descriptions}")
    prompt3 = SystemMessage(
        "Based on the text and the image descriptions, generate 3 questions that can be used to test understanding of the content that feature the images either in the question or the answer. (Specify where the image should be used in the question or answer). Structure your output as clean JSON"
    )
    prompt = [prompt1, prompt2, prompt3]
    chain = model | JsonOutputParser()
    flashcards = chain.invoke(prompt)
    return flashcards


chain = read_pdf | clean_up_text | describe_images | mock_relevant_image_selection | generate_questions

chain.invoke(
    {"pdf_path": "pdf/ML4HLecture02image.pdf", "start_page": 23, "end_page": 26},
    {"run_name": "Generate Image Enriched Flashcards"},
)
