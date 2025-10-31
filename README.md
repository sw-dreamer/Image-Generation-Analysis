
# **Llama + Stable Diffusion + BLIP2 기반 AI 이미지 생성 및 분석 웹 서비스**

이 프로젝트는 **텍스트 입력(한국어/영어)** 을 기반으로:
- 한국어 → Llama 번역 → 영어 프롬프트 → Stable Diffusion → 이미지 생성  
- 영어 → Llama 확장 → 더 자연스럽고 생동감 있는 프롬프트 → Stable Diffusion → 이미지 생성  
을 수행합니다.  
또한 **이미지 + 텍스트 질의**를 입력받아 **BLIP2 모델**을 활용해 이미지 내용을 분석하고 답변을 제공합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 🖼️ 이미지 생성 | 한국어 또는 영어로 입력 시, Llama 모델이 영어 프롬프트를 생성 및 확장 후 Stable Diffusion으로 이미지를 생성합니다. |
| 🔍 이미지 분석 | 사용자가 업로드한 이미지와 텍스트 질문을 기반으로 BLIP2 모델이 답변을 제공합니다. |
| 🌐 웹 인터페이스 | FastAPI 기반의 간단한 HTML UI 제공 |
| 🧩 GPU/CPU 자동 전환 | CUDA 사용 여부에 따라 자동으로 최적화된 실행 모드 선택 |
| ⚙️ Ngrok | 외부 접근 가능한 URL 자동 생성 |

---

## 시스템 아키텍처
```
[사용자 입력]
       │
       ├─ 한국어 → Llama 번역 → 영어 prompt → Stable Diffusion → 이미지 생성
       │
       ├─ 영어 → Llama 확장 → 자연스럽고 직관적인 스타일의 영어 prompt → Stable Diffusion → 이미지 생성
       │
       └─ 이미지 + 텍스트 질문 → BLIP2 → 이미지 내용 분석 및 답변 생성

```

---
## 모델 구성

| 모델 | 역할 | 소스 |
|------|------|------|
| Llama 3.2 3B-Instruct | 번역 및 프롬프트 확장 | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Stable Diffusion v1-5 | 이미지 생성 | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| BLIP2-Flan-T5-XL | 이미지+텍스트 분석 | [Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl) |
