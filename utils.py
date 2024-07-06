import base64
from langchain_core.messages import HumanMessage


def image_to_human_message(text: str, image_path: str) -> HumanMessage:
    image_path = "drag_drop_exercise/images/image.png"
    image_data = base64.b64encode(open(image_path, "rb").read())\
        .decode("utf-8")
    img_url = {"url": f"data:image/png;base64,{image_data}"}
    message = HumanMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": img_url
            },
        ],
        additional_kwargs={"image_path": image_path}
    )
    return message
