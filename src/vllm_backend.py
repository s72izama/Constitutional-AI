
import torch


class VLLMBackend:
    def __init__(
        self,
        model_name_or_path,
        tokenizer,
        cache_dir,
        tensor_parallel_size,
        gpu_memory_utilization,
    ):
        from vllm import LLM
        
        self.tokenizer = tokenizer
        self.model = LLM(
            model=model_name_or_path,
            dtype=torch.bfloat16,
            download_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(self, messages, sampling_params):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        return [outputs[0].outputs[0].text]
