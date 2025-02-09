from openai import OpenAI


class OpenAIModel:
    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 800
    ) -> str:
        return (
            self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            .choices[0]
            .message.content
        )
