import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TextsDataset(Dataset):
    def __init__(self, *paths):
        pd.set_option("future.no_silent_downcasting", True)

        self.texts = pd.concat(
            (pd.read_csv(path) for path in paths),
            axis=0,
            ignore_index=True,
        )
        unique_topics = self.texts.topic.unique().tolist()
        self.texts.topic = self.texts.topic.replace(
            unique_topics, np.arange(len(unique_topics))
        )
        self.texts = self.texts.to_numpy()
        np.random.shuffle(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        label, text = self.texts[idx]
        return text, label, idx

    def get_class(self, idx):
        return self.texts[idx][1]
