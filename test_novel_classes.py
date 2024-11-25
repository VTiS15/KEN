import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import Subset, ConcatDataset

from KEN.metric.KEN import KEN_Evaluator
from KEN.datasets.TextDataset import TextDataset
from KEN.datasets.TextsDataset import TextsDataset


extractors = [
    # {"name": "sentence-transformer", "model": "bert-base-nli-mean-tokens"},
    # {"name": "sentence-transformer", "model": "multi-qa-mpnet-base-cos-v1"},
    # {"name": "sentence-transformer", "model": "all-mpnet-base-v2"},
    # {"name": "sentence-transformer", "model": "stsb-roberta-base"},
    {"name": "openai", "model": "text-embedding-3-large"},
]
tests = [
    # {"kernel": "gaussian", "sigma": 0.1, "eta": 1},
    {"kernel": "gaussian", "sigma": 0.5, "eta": 1},
    {"kernel": "gaussian", "sigma": 1, "eta": 1},
    # {"kernel": "gaussian", "sigma": 5, "eta": 1},
    # {"kernel": "gaussian", "sigma": 1, "eta": 0.1},
    # {"kernel": "gaussian", "sigma": 1, "eta": 0.5},
    {"kernel": "gaussian", "sigma": 1, "eta": 5},
    # {"kernel": "cosine", "sigma": 1, "eta": 0.1},
    # {"kernel": "cosine", "sigma": 1, "eta": 0.5},
    {"kernel": "cosine", "sigma": 1, "eta": 1},
    {"kernel": "cosine", "sigma": 1, "eta": 5},
]


if __name__ == "__main__":
    novel_paths = glob("/home/vtis/KEN-dataset/novel/*.csv")
    ref_path = "~/KEN-dataset/reference.csv"
    num_samples = 1000

    ratios = np.linspace(0.0, 1.0, 11, endpoint=True)
    colors = plt.get_cmap("jet")(np.linspace(0.0, 1.0, len(novel_paths), endpoint=True))

    for extractor in extractors:
        fig, axes = plt.subplots(1, len(tests), figsize=(len(tests) * 5, 5), dpi=600)

        for test_idx, test in enumerate(tests):
            ax = axes[test_idx]

            for num_novel_modes in range(1, len(novel_paths) + 1):
                scores = np.zeros(11)

                for ratio_idx, ratio in enumerate(ratios):
                    num_novel_samples = int(num_samples * ratio)
                    novel_dataset = ConcatDataset(
                        [
                            Subset(
                                TextsDataset(*novel_paths[:num_novel_modes]),
                                range(num_novel_samples),
                            ),
                            Subset(
                                TextDataset("~/KEN-dataset/test-francehistory.csv"),
                                range(num_novel_samples, num_samples),
                            ),
                        ]
                    )
                    ref_dataset = TextDataset(ref_path)

                    KEN = KEN_Evaluator(
                        logger_path="./logs",
                        batchsize=128,
                        **test,
                        num_samples=num_samples,
                        result_name=f"{num_novel_modes}-novel-{num_novel_samples}",
                    )

                    KEN.set_feature_extractor(**extractor, save_path="./save")
                    scores[ratio_idx] = KEN.compute_KEN_with_datasets(
                        novel_dataset,
                        ref_dataset,
                        cholesky_acceleration=True,
                    )

                if test_idx == len(tests) - 1:
                    if num_novel_modes == 1:
                        label = "1 novel class"
                    else:
                        label = f"{num_novel_modes} novel classes"

                    ax.plot(
                        ratios,
                        scores,
                        marker="o",
                        color=colors[num_novel_modes - 1],
                        label=label,
                    )

                    ax.legend(
                        bbox_to_anchor=(1.04, 0.5),
                        loc="center left",
                        borderaxespad=0,
                    )
                else:
                    ax.plot(
                        ratios, scores, marker="o", color=colors[num_novel_modes - 1]
                    )

            ax.set(
                title=f"{test['kernel']}, sigma={test['sigma']}, eta={test['eta']}",
                xlabel="Novel ratio",
                ylabel="KEN score",
                box_aspect=1,
            )
            ax.grid(axis="y")

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.suptitle(f"{extractor['name']}/{extractor['model']}")
        fig.savefig(
            f"plots/{extractor['name']}-{extractor['model']}_modes.png",
            bbox_inches="tight",
        )
        plt.close(fig)
