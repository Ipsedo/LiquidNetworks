# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Tuple

import pandas as pd
import torch as th
from pandarallel import pandarallel
from torch.utils.data import Dataset


# https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
class HouseholdPowerDataset(Dataset):
    def __init__(self, csv_path: str) -> None:
        super().__init__()

        pandarallel.initialize()

        self.__seq_length = 32

        self.__df = pd.read_csv(csv_path, sep=";", header=0, low_memory=False)
        self.__df = self.__df.iloc[len(self.__df) % self.__seq_length :, :]
        self.__df["date"] = self.__df.parallel_apply(
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

        for c in self.__features_column:
            self.__df[c] = self.__df[c].replace("?", "0.0")
            c_data = self.__df[c].dropna().astype(float)
            c_mean = c_data.mean()
            c_std = c_data.std()

            self.__df[c] = self.__df[c].fillna(c_mean).astype(float)

            self.__df[c] = (self.__df[c] - c_mean) / (c_std + 1e-8)

        self.__df[self.__target_variable] = self.__df[
            self.__target_variable
        ].replace("?", "0.0")

        y = self.__df[self.__target_variable].dropna().astype(float)

        y_min = y.min()
        y_max = y.max()
        y_mean = y.mean()

        self.__df[self.__target_variable] = (
            self.__df[self.__target_variable].fillna(y_mean).astype(float)
        )

        self.__df[self.__target_variable] = (
            self.__df[self.__target_variable] - y_min
        ) / (y_max - y_min)

    def __len__(self) -> int:
        return len(self.__df) // self.__seq_length

    def __getitem__(
        self, index: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        index_start = index * self.__seq_length
        index_end = (index + 1) * self.__seq_length

        sub_df = self.__df.iloc[index_start:index_end, :]

        target_variable = sub_df[self.__target_variable]
        features_df = sub_df[self.__features_column]

        # dates = sub_df["date"]
        # deltas = [1.0]
        # for i, d in enumerate(dates[1:]):
        #    deltas.append((d - dates.iloc[i]).total_seconds())

        return (
            th.tensor(features_df.to_numpy().T, dtype=th.float),
            th.ones(self.__seq_length, dtype=th.float),
            th.tensor(target_variable.to_numpy(), dtype=th.float),
        )
