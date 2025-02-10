import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from tenacity import retry, wait_random_exponential


class VertexAIModel:
    def __init__(self, model_id: str):
        vertexai.init(project="fyp-essays", location="us-central1")
        self.model = GenerativeModel(model_id)

    @retry(wait=wait_random_exponential(multiplier=1, max=60))
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 800,
    ) -> str:
        return self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        ).text
