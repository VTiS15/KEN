import os
import argparse
import csv
import torch

from tqdm import tqdm
from dotenv import load_dotenv

from KEN.models import (
    OpenAIModel,
    HuggingFaceModel,
    VertexAIModel,
    AzureOpenAIModel,
)

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--topic",
    type=str,
    required=True,
    help="Topic of essays to generate",
)
parser.add_argument(
    "-n", "--n", type=int, required=True, help="Number of essays to generate"
)
parser.add_argument(
    "-f", "--filename", type=str, required=True, help="Name of file of generated essays"
)
args = parser.parse_args()

# LLMs we use
# llama = HuggingFaceModel("meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.bfloat16)
gpt = AzureOpenAIModel(
    os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), "gpt-4o"
)
# gemini = VertexAIModel("gemini-2.0-flash-001")
# qwen = HuggingFaceModel("Qwen/Qwen2.5-3B-Instruct")
# deepseek = OpenAIModel(
#     os.getenv("OPENROUTER_API_KEY"),
#     os.getenv("OPENROUTER_BASE_URL"),
#     "deepseek/deepseek-chat",
# )

with open(args.filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["topic", "essay"])
    for idx in tqdm(range(args.n), desc="Generating essays"):
        writer.writerow(
            [
                args.topic,
                gpt.generate(
                    f"Generate a one-paragraph essay about {args.topic}.",
                    top_p=0.95,
                ),
            ]
        )
