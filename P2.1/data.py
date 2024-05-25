from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array


class StoreInstance:
    def __init__(self, ID: int, data: np.ndarray):
        self.ID = ID
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def split(
        self, pval: float = 0.0, ptest: float = 0.0
    ) -> Tuple[StoreInstance, StoreInstance, StoreInstance]:
        nval = int(pval * len(self))
        ntest = int(ptest * len(self))
        train = StoreInstance(self.ID, self.data[: -(nval + ntest)])
        val = StoreInstance(self.ID, self.data[-(ntest + nval) : -ntest])
        test = StoreInstance(self.ID, self.data[-ntest:])
        return train, val, test

    def to_tf(
        self,
        seq_len: int,
        batch_size: int,
        norm: bool = False,
        params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        delay = seq_len + WalmartDataset.TARGET_TIMESTEP - 1
        inputs, targets = (
            self.data[:-delay],
            self.data[delay:, WalmartDataset.TARGET_INDEX],
        )
        if norm and params is not None:
            means, stds = params
            inputs = (inputs - means) / stds
            targets = (targets - means[WalmartDataset.TARGET_INDEX]) / stds[
                WalmartDataset.TARGET_INDEX
            ]
        data = timeseries_dataset_from_array(
            inputs,
            targets,
            seq_len,
            sampling_rate=1,
            start_index=0,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return data


class WalmartDataset:
    TARGET_INDEX = 0
    TARGET_TIMESTEP = 3

    def __init__(self, stores: List[StoreInstance], dates: pd.Series):
        self.stores = stores
        self.dates = dates

    def __len__(self) -> int:
        return len(self.stores)

    def __getitem__(self, item: int) -> StoreInstance:
        return self.stores[item]

    def split(
        self, pval: float = 0.0, ptest: float = 0.0
    ) -> Tuple[WalmartDataset, WalmartDataset, WalmartDataset]:
        """Creates a data split of the validation and test set with the most recent samples.

        Args:
            pval (float, optional): Validation ratio. Defaults to 0.0.
            ptest (float, optional): Test ratio. Defaults to 0.0.
        """
        nval = int(pval * len(self.dates))
        ntest = int(ptest * len(self.dates))
        train, val, test = zip(*[store.split(pval, ptest) for store in self.stores])
        return (
            WalmartDataset(train, self.dates[: -(ntest + nval)]),
            WalmartDataset(val, self.dates[-(ntest + nval) : -ntest]),
            WalmartDataset(test, self.dates[-ntest:]),
        )

    def to_tf(
        self,
        seq_len: int,
        batch_size: int = 128,
        shuffle: bool = True,
        norm: bool = True,
        params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> tf.data.Dataset:
        if params is None:
            params = self.means, self.stds
        data = self.stores[0].to_tf(seq_len, batch_size, norm, params, shuffle)
        for store in self.stores[1:]:
            data = data.concatenate(
                store.to_tf(seq_len, batch_size, norm, params, shuffle)
            )
        return data

    def to_numpy(
        self,
        seq_len: int,
        norm: bool = True,
        params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = self.to_tf(seq_len, shuffle=False, norm=norm, params=params)
        inputs, targets = map(np.concatenate, zip(*data.as_numpy_iterator()))
        return inputs, targets

    @property
    def inputs(self) -> np.ndarray:
        return np.concatenate([store.data for store in self.stores], 0)

    @property
    def targets(self) -> np.ndarray:
        return self.inputs[:, self.TARGET_INDEX]

    @property
    def means(self) -> np.ndarray:
        return np.mean(self.inputs, 0)

    @property
    def stds(self) -> np.ndarray:
        return np.std(self.inputs, 0)

    @property
    def target_std(self) -> float:
        return self.stds[self.TARGET_INDEX]

    @classmethod
    def load(cls, path: str) -> WalmartDataset:
        # read adata
        data = pd.read_csv(path)
        data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
        data.index = pd.MultiIndex.from_frame(data[["Store", "Date"]])
        dates = sorted(data["Date"].unique())
        data = data.sort_index().drop(["Store", "Date"], axis=1)

        store_idxs = data.index.get_level_values(0)
        stores = [
            StoreInstance(idx, data[store_idxs == idx].values)
            for idx in store_idxs.unique()
        ]
        return WalmartDataset(stores, dates)
