import os
from typing import Callable, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.data import Dataset
from tensorflow.keras.utils import image_dataset_from_directory

PARTITIONS = ['train', 'val', 'test']

class CelebADataset:
    IMG_SIZE = (64, 64, 3)

    def __init__(self, partition: str, folder: str = 'archive'):
        self.folder = folder
        self.partition = partition
        
    def __len__(self) -> int:
        return len(os.listdir(f'{self.folder}/{self.partition}'))

    def to_tf(
        self,
        norm: Callable,
        batch_size: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        targets: bool = False,
    ) -> Dataset:
        df = image_dataset_from_directory(
            f"{self.folder}/{self.partition}",
            labels=None,
            color_mode="rgb",
            image_size=self.IMG_SIZE[:2],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            interpolation="bilinear",
        )
        df = df.repeat().map(norm)
        if targets:
            df = Dataset.zip(df, df)
        return df
    
    def __getitem__(self, filename: str) -> np.ndarray:
        img = Image.open(f'{self.folder}/{self.partition}/{filename}')
        img = img.resize(self.IMG_SIZE[:2])
        return np.array(img)
    
    @property 
    def imgs(self) -> List[np.ndarray]:
        images = []
        for file in os.listdir(f'{self.folder}/{self.partition}'):
            img = Image.open(f'{self.folder}/{self.partition}/{file}')
            img = img.resize(self.IMG_SIZE[:2])
            images.append(np.array(img))
        return images 
        

    def stream(
        self, norm: Optional[Callable], batch_size: int
    ) -> Iterable[Tuple[List[str], np.ndarray]]:
        files = os.listdir(f"{self.folder}/{self.partition}/")
        for i in range(0, len(files), batch_size):
            batch = []
            for file in files[i : (i + batch_size)]:
                img = Image.open(f"{self.folder}/{self.partition}/{file}")
                img = img.resize(self.IMG_SIZE[:2])
                img = np.array(img)
                batch.append(norm(img) if norm is not None else img)
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
