import re
from os import listdir
from os.path import isdir, isfile, join

import pandas as pd
import torch as th

from liquid_networks import networks

from ..abstract_dataset import AbstractDataset, AbstractDatasetFactory


# MotionSense Dataset: Sensor Based Human Activity and Attribute Recognition
class MotionSenseDataset(AbstractDataset[th.Tensor]):
    def __init__(self, dataset_path: str, load_train: bool, sequence_length: int) -> None:
        # pylint: disable=too-many-locals
        super().__init__(dataset_path)

        train_trials = list(range(1, 10))

        self.__seq_length = sequence_length

        regex_activity = re.compile(r"^(\w+)_(\d+)$")
        regex_subject = re.compile(r"^sub_(\d+)\.csv$")

        data_path = join(dataset_path, "A_DeviceMotion_data", "A_DeviceMotion_data")

        df_list: list[pd.DataFrame] = []

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
                        sub_df = sub_df.iloc[len(sub_df) % self.__seq_length :, :]
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
            c: i for i, c in enumerate(sorted(self.__df[self.__target_column].unique()))
        }

    def __len__(self) -> int:
        return len(self.__df) // self.__seq_length

    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]:
        index_start = index * self.__seq_length
        index_end = (index + 1) * self.__seq_length

        sub_df = self.__df.iloc[index_start:index_end]

        features_df = sub_df[self.__features_columns].astype(float).fillna(0)
        target_variable = self.__class_to_idx[sub_df[self.__target_column].iloc[0]]

        return th.tensor(features_df.to_numpy(), dtype=th.float), th.tensor(
            target_variable, dtype=th.long
        )

    @property
    def task_type(self) -> networks.TaskType:
        return networks.TaskType.LAST_CLASSIFICATION

    def to_device(self, data: th.Tensor, device: th.device) -> th.Tensor:
        return data.to(device)


class MotionSenseDatasetFactory(AbstractDatasetFactory[th.Tensor]):
    def get_dataset(self, data_path: str) -> AbstractDataset[th.Tensor]:
        return MotionSenseDataset(
            data_path,
            self._get_config("load_train", bool, True),
            self._get_config("sequence_length", int, 32),
        )
