from KEN.metric.KEN import KEN_Evaluator
from KEN.datasets.TextDataset import TextDataset


if __name__ == "__main__":
    num_samples = 1000
    dataset1 = TextDataset("~/KEN-dataset/gpt-4o_universe.csv")
    dataset2 = TextDataset("~/KEN-dataset/gemini-2.0_universe.csv")
    model_id = "sentence-transformers/all-mpnet-base-v2"

    KEN = KEN_Evaluator(
        logger_path="./logs",
        batchsize=128,
        kernel="gaussian",
        sigma=1,
        eta=5,
        num_samples=num_samples,
        result_name=f"gemini-2.0_gpt-4o_{model_id.replace("/", "_")}",
    )

    KEN.set_feature_extractor(*model_id.split("/"), save_path="./save")
    KEN.compute_KEN_with_datasets(
        dataset2,
        dataset1,
        cholesky_acceleration=True,
    )
