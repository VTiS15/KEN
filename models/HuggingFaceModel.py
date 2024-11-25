import torch
from transformers import pipeline


class HuggingFaceModel:
    def __init__(self, model_id: str, torch_dtype: str | torch.dtype = "auto"):
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 800
    ) -> str:
        return self.pipe(
            [
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            return_full_text=False,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )