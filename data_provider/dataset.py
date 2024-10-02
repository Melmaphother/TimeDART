import numpy as np
import pandas as pd
import torch
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ETThDataset(Dataset):
    def __init__(self, args: Namespace, flag: str = "train", dataset: str = None):
        """
        Constructor of the ETTDataset class, including ETTh1, ETTh2, ETTm1, ETTm2.
        Args:
            args: args of whole project
            flag: 'train', 'val' or 'test'
        """
        match flag:
            case "train" | "val" | "test":
                pass
            case _:
                raise ValueError(f"Invalid flag: {flag=}")

        self.args = args
        self.flag = flag
        if dataset is not None:
            # when mixing datasets, the dataset name is provided
            self.file_name = dataset
        else:
            self.file_name = args.dataset
        if args.verbose:
            print(f"Loading {self.file_name} dataset... for {self.flag}")
        self.scale = args.scale
        self.scaler = StandardScaler()
        self.slice_map = {
            "train": slice(0, 12 * 30 * 24),
            "val": slice(
                12 * 30 * 24 - self.args.input_len, 12 * 30 * 24 + 4 * 30 * 24
            ),
            "test": slice(
                12 * 30 * 24 + 4 * 30 * 24 - self.args.input_len,
                12 * 30 * 24 + 8 * 30 * 24,
            ),
            # sampling interval = 1 hour
            # train: 12 months = [0, 8639) samples = 8640 samples
            # val: 4 months = [8640, 11520) samples = 2880 samples
            # test: 4 months = [11520, 14400) samples = 2880 samples
        }
        self.__read_data()

    def __read_data(self):
        file_path = Path(self.args.data_dir) / "ETT-small" / f"{self.file_name}.csv"
        ett_data = pd.read_csv(file_path, index_col="date", parse_dates=True)
        ett_data = ett_data.to_numpy(dtype=np.float32)

        if self.scale:
            train_slice = self.slice_map["train"]
            self.scaler.fit(ett_data[train_slice])
            ett_data = self.scaler.transform(ett_data)

        _slice = self.slice_map[self.flag]
        self.data = ett_data[_slice]

    def __len__(self):
        return len(self.data) - self.args.input_len - self.args.pred_len + 1

    def __getitem__(self, idx):
        input_begin = idx
        input_end = idx + self.args.input_len
        label_begin = input_end
        label_end = input_end + self.args.pred_len

        x = self.data[input_begin:input_end]
        y = self.data[label_begin:label_end]

        x = torch.tensor(x, dtype=torch.float32).to(self.args.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.args.device)

        return x, y


class ETTmDataset(Dataset):
    def __init__(self, args: Namespace, flag: str = "train", dataset: str = None):
        """
        Constructor of the ETTDataset class, including ETTh1, ETTh2, ETTm1, ETTm2.
        Args:
            args: args of whole project
            flag: 'train', 'val' or 'test'
        """
        match flag:
            case "train" | "val" | "test":
                pass
            case _:
                raise ValueError(f"Invalid flag: {flag=}")

        self.args = args
        self.flag = flag
        if dataset is not None:
            # when mixing datasets, the dataset name is provided
            self.file_name = dataset
        else:
            self.file_name = args.dataset
        if args.verbose:
            print(f"Loading {self.file_name} dataset... for {self.flag}")
        self.scale = args.scale
        self.scaler = StandardScaler()
        self.slice_map = {
            "train": slice(0, 12 * 30 * 24 * 4),
            "val": slice(
                12 * 30 * 24 * 4 - self.args.input_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            ),
            "test": slice(
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.args.input_len,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ),
            # sampling interval = 15 minutes
            # train: 12 months = [0, 34559) samples = 34560 samples
            # val: 4 months = [34560, 46080) samples = 11520 samples
            # test: 4 months = [46080, 57600) samples = 11520 samples
        }
        self.__read_data()

    def __read_data(self):
        file_path = Path(self.args.data_dir) / "ETT-small" / f"{self.file_name}.csv"
        ett_data = pd.read_csv(file_path, index_col="date", parse_dates=True)
        ett_data = ett_data.to_numpy(dtype=np.float32)

        if self.scale:
            train_slice = self.slice_map["train"]
            self.scaler.fit(ett_data[train_slice])
            ett_data = self.scaler.transform(ett_data)

        _slice = self.slice_map[self.flag]
        self.data = ett_data[_slice]

    def __len__(self):
        return len(self.data) - self.args.input_len - self.args.pred_len + 1

    def __getitem__(self, idx):
        input_begin = idx
        # if self.flag == "train":
        #     if idx % 2 == 1:
        #         input_begin = idx - 1

        input_end = idx + self.args.input_len
        label_begin = input_end
        label_end = input_end + self.args.pred_len

        x = self.data[input_begin:input_end]
        y = self.data[label_begin:label_end]

        x = torch.tensor(x, dtype=torch.float32).to(self.args.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.args.device)

        return x, y


class CustomDataset(Dataset):
    def __init__(
        self,
        args: Namespace,
        flag: str = "train",
        dataset: str = None,
    ):
        self.args = args
        self.flag = flag
        match flag:
            case "train" | "val" | "test":
                pass
            case _:
                raise ValueError(f"Invalid flag: {flag=}")
        if dataset is not None:
            # when mixing datasets, the dataset name is provided
            self.file_name = dataset
        else:
            self.file_name = args.dataset
        if self.args.verbose:
            print(f"Loading {self.file_name} dataset... for {self.flag}")
        self.scale = args.scale
        self.scaler = StandardScaler()
        self.__read_data()

    def __read_data(self):
        file_path = Path(self.args.data_dir) / self.file_name / f"{self.file_name}.csv"
        _data = pd.read_csv(file_path, index_col="date", parse_dates=True)
        train_samples_len = int(len(_data) * 0.7)
        test_samples_len = int(len(_data) * 0.2)
        val_samples_len = len(_data) - train_samples_len - test_samples_len
        self.slice_map = {
            "train": slice(0, train_samples_len),
            "val": slice(
                train_samples_len - self.args.input_len,
                train_samples_len + val_samples_len,
            ),
            "test": slice(
                len(_data) - test_samples_len - self.args.input_len, len(_data)
            ),
        }

        if self.scale:
            train_slice = self.slice_map["train"]
            self.scaler.fit(_data[train_slice])
            _data = self.scaler.transform(_data)

        _slice = self.slice_map[self.flag]
        self.data = _data[_slice]

    def __len__(self):
        return len(self.data) - self.args.input_len - self.args.pred_len + 1

    def __getitem__(self, idx):
        input_begin = idx
        # if self.flag == "train":
        #     if idx % 2 == 1:
        #         input_begin = idx - 1

        input_end = idx + self.args.input_len
        label_begin = input_end
        label_end = input_end + self.args.pred_len

        x = self.data[input_begin:input_end]
        y = self.data[label_begin:label_end]

        x = torch.tensor(x, dtype=torch.float32).to(self.args.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.args.device)

        return x, y
