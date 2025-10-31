import gc
import time
import torch

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def wait_for_gpu_memory(required_mb=1000, check_interval=5):
    if not torch.cuda.is_available():
        return
    while True:
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 2)
        if free_mem >= required_mb:
            break
        print(f"[GPU 메모리 부족] 현재 여유: {free_mem:.2f} MB, {check_interval}s 후 재시도")
        time.sleep(check_interval)
