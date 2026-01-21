import torch
from transformers import AutoModelForCausalLM


class HFBackend:
    def __init__(
        self,
        model_name_or_path,
        tokenizer,
        device,
        cache_dir,
    ):
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            cache_dir=cache_dir,
        ).to(device)

        self.model.eval()

    def generate(self, messages, sampling_params):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
                repetition_penalty=sampling_params.repetition_penalty,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return [self.tokenizer.decode(output_ids[0], skip_special_tokens=True)]
