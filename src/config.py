import os
import torch
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Config")

# 환경 변수
os.environ['HF_TOKEN'] = 'Llama-3.2-3B-Instruct 토큰'
os.environ['MODEL_ID_LLAMA'] = 'meta-llama/Llama-3.2-3B-Instruct'
os.environ['MODEL_ID_SD'] = 'runwayml/stable-diffusion-v1-5'
os.environ['NGROK_AUTH_TOKEN'] = 'Grok 토큰'

HF_TOKEN = os.environ['HF_TOKEN']
MODEL_ID_LLAMA = os.environ['MODEL_ID_LLAMA']
MODEL_ID_SD = os.environ['MODEL_ID_SD']
NGROK_AUTH_TOKEN = os.environ['NGROK_AUTH_TOKEN']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

