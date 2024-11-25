from openai import AzureOpenAI


class OpenAIModel:
    def __init__(self, azure_endpoint: str, api_key: str, model_id: str):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-10-21",
        )
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    },
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            .choices[0]
            .message.content
        )
