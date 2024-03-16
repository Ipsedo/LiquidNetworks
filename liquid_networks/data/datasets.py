# -*- coding: utf-8 -*-
import re
from abc import ABC, abstractmethod
from datetime import datetime
from os import listdir
from os.path import isdir, isfile, join
from typing import List, Tuple

import pandas as pd
import torch as th
from torch.utils.data import Dataset
from tqdm import tqdm

from .. import networks
from .transform import min_max_normalize_column, standardize_column


class AbstractDataset(ABC, Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()

        self._data_path = data_path

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        pass

    @property
    @abstractmethod
    def task_type(self) -> networks.TaskType:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(data_path={self._data_path})"

    def __repr__(self) -> str:
        return str(self)


# https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
class HouseholdPowerDataset(AbstractDataset):
    def __init__(self, csv_path: str) -> None:
        # pylint: disable=too-many-locals
        super().__init__(csv_path)

        self.__seq_length = 32

        self.__df = pd.read_csv(csv_path, sep=";", header=0, low_memory=False)
        self.__df = self.__df.iloc[len(self.__df) % self.__seq_length :, :]
        self.__df["date"] = self.__df.apply(
            lambda r: datetime.strptime(
                f"{r['Date']} {r['Time']}", "%d/%m/%Y %H:%M:%S"
            ),
            axis=1,
        )
        self.__df = self.__df.drop(columns=["Date", "Time"])

        self.__target_variable = "Global_active_power"
        self.__features_column = [
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]

        # pre-process features
        for c in self.__features_column:
            self.__df[c] = standardize_column(self.__df[c].replace("?", "0.0"))

        # pre-process target

        self.__df[self.__target_variable] = standardize_column(
            self.__df[self.__target_variable].replace("?", "0.0")
        )

        # pre-process time deltas
        dates = self.__df["date"].tolist()
        deltas = [(dates[1] - dates[0]).total_seconds()]
        for i, d in enumerate(dates[1:]):
            deltas.append((d - dates[i]).total_seconds())

        self.__df["delta"] = min_max_normalize_column(pd.Series(deltas))

    def __len__(self) -> int:
        return len(self.__df) // self.__seq_length

    def __getitem__(
        self, index: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        index_start = index * self.__seq_length
        index_end = (index + 1) * self.__seq_length

        sub_df = self.__df.iloc[index_start:index_end, :]

        target_variable = sub_df[[self.__target_variable]]
        features_df = sub_df[self.__features_column]
        delta = sub_df["delta"]

        return (
            th.tensor(features_df.to_numpy().T, dtype=th.float),
            th.tensor(delta.to_numpy(), dtype=th.float),
            th.tensor(target_variable.to_numpy().T, dtype=th.float),
        )

    @property
    def task_type(self) -> networks.TaskType:
        return "regression"


# MotionSense Dataset: Sensor Based Human Activity and Attribute Recognition
class MotionSenseDataset(AbstractDataset):
    def __init__(self, dataset_path: str, load_train: bool = True) -> None:
        # pylint: disable=too-many-locals
        super().__init__(dataset_path)

        train_trials = list(range(1, 10))

        self.__seq_length = 32

        regex_activity = re.compile(r"^(\w+)_(\d+)$")
        regex_subject = re.compile(r"^sub_(\d+)\.csv$")

        data_path = join(
            dataset_path, "A_DeviceMotion_data", "A_DeviceMotion_data"
        )

        df_list: List[pd.DataFrame] = []

        for d in listdir(data_path):
            dir_path = join(data_path, d)
            matched_dir = regex_activity.match(d)
            if isdir(dir_path) and matched_dir:
                act = matched_dir.group(1)
                trial = matched_dir.group(2)

                for f in listdir(dir_path):
                    matched_file = regex_subject.match(f)
                    if isfile(join(dir_path, f)) and matched_file:
                        subject = matched_file.group(1)

                        sub_df = pd.read_csv(join(dir_path, f), sep=",")
                        sub_df["act"] = act
                        sub_df["trial"] = int(trial)
                        sub_df["subject"] = int(subject)
                        sub_df["file_index"] = len(df_list)
                        sub_df = sub_df.iloc[
                            len(sub_df) % self.__seq_length :, :
                        ]
                        sub_df["time"] = list(range(len(sub_df)))

                        df_list.append(sub_df)

        self.__df = pd.concat(df_list).drop("Unnamed: 0", axis=1)

        cond = self.__df["trial"].isin(train_trials)
        self.__df = self.__df[cond if load_train else ~cond]

        self.__features_columns = [
            "attitude.roll",
            "attitude.pitch",
            "attitude.yaw",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
        ]
        self.__target_column = "act"

        self.__class_to_idx = {
            c: i
            for i, c in enumerate(
                sorted(self.__df[self.__target_column].unique())
            )
        }

        for c in self.__features_columns:
            self.__df[c] = standardize_column(self.__df[c])

    def __len__(self) -> int:
        return len(self.__df) // self.__seq_length

    def __getitem__(
        self, index: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        index_start = index * self.__seq_length
        index_end = (index + 1) * self.__seq_length

        sub_df = self.__df.iloc[index_start:index_end]

        features_df = sub_df[self.__features_columns].astype(float).fillna(0)
        target_variable = self.__class_to_idx[
            sub_df[self.__target_column].iloc[0]
        ]

        return (
            th.tensor(features_df.to_numpy().T, dtype=th.float),
            th.ones(len(features_df), dtype=th.float),
            th.tensor(target_variable, dtype=th.long),
        )

    @property
    def task_type(self) -> networks.TaskType:
        return "last_classification"


class HarmfulBrainActivityDataset(AbstractDataset):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)

        assert isdir(data_path)

        re_file = re.compile(r"^\d+_eeg\.pt$")

        self.__index_list = [
            f.split("_")[0]
            for f in tqdm(listdir(data_path))
            if isfile(join(data_path, f)) and re_file.match(f)
        ]

    def __len__(self) -> int:
        return len(self.__index_list)

    def __getitem__(
        self, index: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        file_index = self.__index_list[index]
        features = th.load(join(self._data_path, f"{file_index}_eeg.pt"))
        return (
            features,
            th.ones(features.size(1), dtype=th.float),
            th.load(join(self._data_path, f"{file_index}_classes.pt")),
        )

    @property
    def task_type(self) -> networks.TaskType:
        return "brain_activity"
