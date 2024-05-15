import os
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.data import Dataset
from tensorflow.keras.utils import image_dataset_from_directory

PARTITIONS = ["train", "val", "test"]


class CelebADataset:
    FOLDER = "archive"
    IMG_SIZE = (128, 128, 3)

    def __init__(self, partition: str):
        assert partition in PARTITIONS, f"Partition {partition} not available"
        self.partition = partition

    def to_tf(
        self,
        norm: Callable,
        batch_size: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        targets: bool = False,
    ) -> Dataset:
        df = image_dataset_from_directory(
            f"{self.FOLDER}/{self.partition}",
            labels=None,
            color_mode="rgb",
            image_size=self.IMG_SIZE[:2],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            interpolation="bilinear",
        )
        df = df.map(norm)
        if targets:
            df = Dataset.zip(df, df)
        return df

    def stream(
        self, norm: Callable, batch_size: int
    ) -> Iterable[Tuple[List[str], np.ndarray]]:
        files = os.listdir(f"{self.FOLDER}/{self.partition}/")
        for i in range(0, len(files), batch_size):
            batch = []
            for file in files[i : (i + batch_size)]:
                img = Image.open(f"{self.FOLDER}/{self.partition}/{file}")
                img = img.resize(self.IMG_SIZE[:2])
                img = np.array(img)
                batch.append(norm(img))
            yield files[i : (i + batch_size)], np.stack(batch)


if __name__ == "__main__":
    # create the train, val and test splits from archive.zip
    assert os.path.exists(
        "archive.zip"
    ), "Could not find archive.zip. Download it at https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
    DATA_PATH = "archive"
    if not os.path.exists(DATA_PATH):
        os.system(f"unzip archive.zip -d {DATA_PATH}")
    partition = pd.read_csv(f"{DATA_PATH}/list_eval_partition.csv")
    for name in PARTITIONS:
        if not os.path.exists(f"{DATA_PATH}/{name}/"):
            os.makedirs(f"{DATA_PATH}/{name}")

    for _, (img, part) in partition.iterrows():
        os.rename(
            f"{DATA_PATH}/img_align_celeba/img_align_celeba/{img}",
            f"{DATA_PATH}/{PARTITIONS[part]}/{img}",
        )
