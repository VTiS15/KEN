from KEN.metric.KEN import KEN_Evaluator
from KEN.datasets.TextDataset import TextDataset


if __name__ == "__main__":
    num_samples = 1000
    test_llm = "deepseek-v3"
    ref_llm = "qwen-2.5-3B"
    embedding = "sentence-transformers/all-mpnet-base-v2"

    KEN = KEN_Evaluator(
        logger_path="./logs",
        batchsize=128,
        kernel="gaussian",
        sigma=1,
        eta=5,
        num_samples=num_samples,
        result_name=f"{test_llm}_{ref_llm}_{embedding.replace("/", "_")}",
    )

    test_dataset = TextDataset(f"datasets/{test_llm}_universe.csv")
    ref_dataset = TextDataset(f"datasets/{ref_llm}_universe.csv")

    KEN.set_feature_extractor(*embedding.split("/"), save_path="./save")
    KEN.compute_KEN_with_datasets(
        test_dataset,
        ref_dataset,
        retrieve_mode=True,
        retrieve_mode_from_both_sets=True,
    )
