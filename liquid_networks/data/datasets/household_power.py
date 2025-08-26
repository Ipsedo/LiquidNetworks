import pandas as pd
import torch as th

from liquid_networks import networks

from ..abstract_dataset import AbstractDataset, AbstractDatasetFactory
from ..prediction_register import AbstractPredictionRegister, NoPredictionRegister


# https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
class HouseholdPowerDataset(AbstractDataset[th.Tensor]):
    def __init__(self, csv_path: str, sequence_length: int) -> None:
        # pylint: disable=too-many-locals
        super().__init__(csv_path)

        self.__seq_length = sequence_length

        self.__df = pd.read_csv(csv_path, sep=";", header=0, low_memory=False)
        self.__df = self.__df.iloc[len(self.__df) % self.__seq_length :, :]
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

    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]:
        index_start = index * self.__seq_length
        index_end = (index + 1) * self.__seq_length

        sub_df = self.__df.iloc[index_start : index_end + 1, :]

        features_df = sub_df[self.__features_column].iloc[:-1, :]
        target_variable = sub_df[[self.__target_variable]].iloc[1:, :]

        return (
            th.tensor(features_df.to_numpy(), dtype=th.float),
            th.tensor(target_variable.to_numpy(), dtype=th.float),
        )

    @property
    def task_type(self) -> networks.TaskType:
        return networks.TaskType.POSITIVE_REGRESSION

    def to_device(self, data: th.Tensor, device: th.device) -> th.Tensor:
        return data.to(device)

    def get_prediction_register(self) -> AbstractPredictionRegister:
        return NoPredictionRegister()


class HouseholdPowerDatasetFactory(AbstractDatasetFactory[th.Tensor]):
    def get_dataset(self, data_path: str) -> AbstractDataset[th.Tensor]:
        return HouseholdPowerDataset(data_path, self._get_config("sequence_length", int, 256))
