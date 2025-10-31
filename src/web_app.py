from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langdetect import detect
from .translator import LlamaTranslator
from .image_generator import ImageGenerator
from .config import MODEL_ID_LLAMA, HF_TOKEN
import logging
from fastapi import FastAPI, UploadFile, File, Form
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import base64


logger = logging.getLogger("Web_App")

app = FastAPI(title="AI Image Generator API", version="1.0")

translator = LlamaTranslator(model_id=MODEL_ID_LLAMA, hf_token=HF_TOKEN)
generator = ImageGenerator()

MODEL_ID_BLIP2 = "Salesforce/blip2-flan-t5-xl"  # 공개 모델
blip_processor = Blip2Processor.from_pretrained(MODEL_ID_BLIP2)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID_BLIP2,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto"
)

class PromptRequest(BaseModel):
    prompt_text: str

@app.get("/generate", response_class=HTMLResponse)
def generate_form():
    return """
    <html>
        <head><title>AI Image Generator</title></head>
        <body>
            <h2>AI Image Generator</h2>
            <form id="generate-form">
                <label>Prompt:</label><br>
                <input type="text" id="prompt_text" size="50"><br><br>
                <button type="button" onclick="submitPrompt()">Generate Image</button>
            </form>
            <h3>Result:</h3>
            <div id="result"></div>
            <script>
                async function submitPrompt() {
                    const prompt = document.getElementById("prompt_text").value;
                    const response = await fetch("/api/generate", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({prompt_text: prompt})
                    });
                    const data = await response.json();
                    document.getElementById("result").innerHTML =
                        "<b>Translated Prompt:</b> " + data.translated_prompt + "<br>" +
                        "<b>Image Path:</b> <a href='" + data.image_path + "' target='_blank'>View Image</a>";
                }
            </script>
        </body>
    </html>
    """

@app.post("/api/generate")
def generate_image_api(req: PromptRequest):
    prompt_text = req.prompt_text
    try:
        lang = detect(prompt_text)
    except Exception:
        lang = "unknown"

    logger.debug(f"입력 프롬프트: {prompt_text}, 언어 감지: {lang}")

    if lang == "ko":
        system_message = (
            "You are a professional prompt engineer and creative translator. "
            "Translate Korean descriptions into vivid English prompts suitable for Stable Diffusion."
        )
    else:
        system_message = (
            "You are a professional Stable Diffusion prompt engineer. "
            "Enhance English prompts into cinematic, visually rich versions suitable for high-quality wallpapers."
        )

    translated_prompt = translator.generate(system_message, prompt_text)
    image, file_path = generator.generate_image(translated_prompt)

    return {
        "original_prompt": prompt_text,
        "translated_prompt": translated_prompt,
        "image_path": file_path
    }
@app.get("/analyze", response_class=HTMLResponse)
def analyze_form():
    html_content = """
    <html>
        <head><title>AI Image + Text Analyzer</title></head>
        <body>
            <h2>AI Image + Text Analyzer</h2>
            <form id="analyze-form" enctype="multipart/form-data">
                <label>Text Input:</label><br>
                <input type="text" id="text_input" size="50"><br><br>
                <label>Upload Image:</label><br>
                <input type="file" id="image_input" accept="image/*"><br><br>
                <button type="button" onclick="submitAnalysis()">Analyze</button>
            </form>
            <h3>Result:</h3>
            <div id="result"></div>
            <script>
                async function submitAnalysis() {
                    const text = document.getElementById("text_input").value;
                    const imageFile = document.getElementById("image_input").files[0];

                    if (!text || !imageFile) {
                        alert("Please provide both text and an image.");
                        return;
                    }

                    const formData = new FormData();
                    formData.append("text", text);
                    formData.append("image", imageFile);

                    const response = await fetch("/api/analyze", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    document.getElementById("result").innerHTML =
                        "<b>Text Input:</b> " + data.text_input + "<br>" +
                        "<b>Answer:</b> " + data.answer + "<br>" +
                        "<b>Uploaded Image:</b><br>" +
                        "<img src='" + data.image_data + "' width='300'/>";

                }
            </script>
        </body>
    </html>
    """
    return html_content


@app.post("/api/analyze")
async def analyze_image_with_text(text: str = Form(...), image: UploadFile = File(...)):
    """
    텍스트 질문 + 이미지 입력 → BLIP2 모델로 답변 생성
    """
    image_bytes = await image.read()
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # BLIP2 입력 생성
    inputs = blip_processor(images=img, text=text, return_tensors="pt").to(DEVICE)

    # 답변 생성
    generated_ids = blip_model.generate(**inputs, max_new_tokens=128)
    answer = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

    # 이미지 Base64 인코딩
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_data_uri = f"data:image/png;base64,{img_str}"

    return {
        "text_input": text,
        "answer": answer,
        "image_data": img_data_uri  # HTML에서 바로 표시 가능
    }


