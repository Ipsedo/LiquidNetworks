from datetime import datetime

import pandas as pd
import torch as th

from liquid_networks import networks

from ..abstract_dataset import AbstractDataset


# https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
class HouseholdPowerDataset(AbstractDataset):
    def __init__(self, csv_path: str) -> None:
        # pylint: disable=too-many-locals
        super().__init__(csv_path)

        self.__seq_length = 256

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

        self.__df = self.__df[self.__df[self.__target_variable] != "?"]

        self.__df[self.__target_variable] = (
            self.__df[self.__target_variable].astype(float).fillna(0.0)
        )
        for c in self.__features_column:
            self.__df[c] = self.__df[c].astype(float).fillna(0.0)

    def __len__(self) -> int:
        return len(self.__df) // self.__seq_length

    def __getitem__(
        self, index: int
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        index_start = index * self.__seq_length
        index_end = (index + 1) * self.__seq_length

        sub_df = self.__df.iloc[index_start : index_end + 1, :]

        features_df = sub_df[self.__features_column].iloc[:-1, :]
        target_variable = sub_df[[self.__target_variable]].iloc[1:, :]

        return (
            th.tensor(features_df.to_numpy(), dtype=th.float),
            th.ones(index_end - index_start, dtype=th.float),
            th.tensor(target_variable.to_numpy(), dtype=th.float),
        )

    @property
    def task_type(self) -> networks.TaskType:
        return "positive_regression"
