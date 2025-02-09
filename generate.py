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
    AzureAIModel,
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

with open(args.filename, "w", newline="") as file:
    model = HuggingFaceModel("Qwen/Qwen2.5-3B-Instruct")
    # model = AzureAIModel(os.getenv("DEEPSEEK_ENDPOINT"), os.getenv("DEEPSEEK_API_KEY"))
    # model = VertexAIModel("gemini-2.0-flash-001")

    writer = csv.writer(file)
    writer.writerow(["topic", "essay"])
    for idx in tqdm(range(args.n), desc="Generating essays"):
        writer.writerow(
            [
                args.topic,
                model.generate(
                    f"Generate a one-paragraph essay about {args.topic}.",
                    top_p=0.95,
                ),
            ]
        )
