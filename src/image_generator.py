import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
from .utils import clear_memory
from .config import MODEL_ID_SD, HF_TOKEN
from PIL import Image
import logging

logger = logging.getLogger("Image_Generator")

class ImageGenerator:
    def __init__(self):
        

        # GPU 사용 가능 시 device_map 설정
        device_map = "balanced" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Stable Diffusion 파이프라인 로드
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID_SD,
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_safetensors=True,
            device_map=device_map,
            token=HF_TOKEN
        )

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()

        if not torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()

        logger.info("[ImageGenerator] Stable Diffusion pipeline 로드 완료")

    def generate_image(self, prompt, height=512, width=512, guidance_scale=8):
        """
        이미지 생성
        GPU: torch.autocast("cuda") 사용
        CPU: torch.autocast 사용하지 않고 생성
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clear_memory()  # 메모리 정리

        if device == "cuda":
            # GPU 환경: autocast 적용
            with torch.autocast("cuda"):
                result = self.pipe(prompt, height=height, width=width, guidance_scale=guidance_scale)
        else:
            # CPU 환경: autocast 사용 불가
            result = self.pipe(prompt, height=height, width=width, guidance_scale=guidance_scale)

        image = result.images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"generated_image_{timestamp}.jpg"
        image.save(file_name)

        logger.debug(f"[ImageGenerator] Image saved to {file_name}")
        return image, file_name


