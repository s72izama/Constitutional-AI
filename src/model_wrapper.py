import torch
from transformers import AutoTokenizer
import platform
from hf_backend import HFBackend
from vllm_backend import VLLMBackend

def is_windows() -> bool:
    return platform.system().lower().startswith("win")

class ModelWrapper:
    def __init__(
        self,
        model_name_or_path,
        cache_dir="./cache",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,
        )

        if is_windows():
            self.backend = HFBackend(
                model_name_or_path,
                self.tokenizer,
                device,
                cache_dir,
            )
        else:
            self.backend = VLLMBackend(
                model_name_or_path,
                self.tokenizer,
                cache_dir,
                tensor_parallel_size,
                gpu_memory_utilization,
            )

    def generate(self, messages, sampling_params):
        return self.backend.generate(messages, sampling_params)

