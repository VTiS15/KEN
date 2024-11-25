import os
import argparse
import csv
import torch

from tqdm import tqdm
from dotenv import load_dotenv

from KEN.models import OpenAIModel, HuggingFaceModel

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
    model = HuggingFaceModel("meta-llama/Llama-3.2-3B-Instruct", torch.bfloat16)
    writer = csv.writer(file)

    writer.writerow(["model", "topic", "essay"])
    for idx in tqdm(range(args.n), desc="Generating essays"):
        writer.writerow(
            [
                args.topic,
                model.generate(
                    f"Generate a one-paragraph essay about {args.topic}.",
                    top_p=0.95,
                )[0]["generated_text"],
            ]
        )
