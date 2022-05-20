import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl

from typing import Union, Dict


def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min']) + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min']) + 1e-5
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method="min_max", norm_statistic=None, interval=1):
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size
        train_data = self.data[lo: hi]
        target_data = self.data[hi:hi + self.horizon]
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx


class ForecastDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data: Union[pd.DataFrame, np.ndarray], 
            batch_size: int = 32,
            window_size: int = 10, 
            horizon: int = 1, 
            train_ratio: float = 0.8,
            valid_test_ratio: float = 0.7,
            normalize_method: str = "min_max",
            interval: int = 1,
            num_workers: int = 4,
        ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
        else:
            self.data = data

        # Split data
        train_ratio = train_ratio if train_ratio < 1 else 1
        valid_ratio = (1 - train_ratio) * valid_test_ratio
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]

        # Normalize data
        if normalize_method.lower() == "z_score":
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        elif normalize_method == 'min_max':
            train_min = np.min(train_data, axis=0)
            train_max = np.max(train_data, axis=0)
            normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
        else:
            raise ValueError(f"Unsupported normalize method: {normalize_method}")

        # Create datasets
        self.train_dataset = ForecastDataset(train_data, window_size, horizon, normalize_method, normalize_statistic, interval)
        self.valid_dataset = ForecastDataset(valid_data, window_size, horizon, normalize_method, normalize_statistic, interval)
        self.test_dataset = ForecastDataset(test_data, window_size, horizon, normalize_method, normalize_statistic, interval)

    def train_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            drop_last=False, 
            shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.valid_dataset, 
            batch_size=self.hparams.batch_size, 
            drop_last=False, 
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            drop_last=False, 
            shuffle=False,
            num_workers=self.hparams.num_workers
        )