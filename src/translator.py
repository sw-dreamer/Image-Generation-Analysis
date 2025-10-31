import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import clear_memory, wait_for_gpu_memory
from .config import DEVICE

logger = logging.getLogger("Translator")

class LlamaTranslator:
    def __init__(self, model_id, hf_token, offload_folder="./llama_offload"):
        self.model_id = model_id
        self.hf_token = hf_token
        self.offload_folder = offload_folder

        clear_memory()
        if DEVICE == "cuda":
            wait_for_gpu_memory(1000)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                device_map="auto" if DEVICE == "cuda" else {"": "cpu"},
                offload_folder=self.offload_folder,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                trust_remote_code=True
            )
        except RuntimeError:
            logger.warning("[경고] GPU 메모리 부족, CPU 오프로드로 재시도...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                device_map={"": "cpu"},
                offload_folder=self.offload_folder,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        logger.info("[LlamaTranslator] 모델 로드 완료")

    def generate(self, system_message, user_message, max_new_tokens=256):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            input_ids = inputs
            attention_mask = torch.ones_like(input_ids)

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [self.tokenizer.eos_token_id]
        if eot_id:
            terminators.append(eot_id)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tuple(terminators) if len(terminators) > 1 else terminators[0],
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        output_ids = outputs[0, input_ids.shape[1]:]
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        logger.debug(f"[번역 결과] {decoded[:200]}")
        return decoded

