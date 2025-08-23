import re
from os import listdir
from os.path import isdir, isfile, join

import torch as th
from tqdm import tqdm

from liquid_networks import networks

from ..abstract_dataset import AbstractDataset


class HarmfulBrainActivityDataset(AbstractDataset[th.Tensor]):
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

    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]:
        file_index = self.__index_list[index]

        # maybe broken, need update notebook
        # new shape: (Batch, Time, Features)
        features = th.abs(
            th.load(join(self._data_path, f"{file_index}_eeg.pt"))
        )
        features = (features - th.mean(features, dim=1, keepdim=True)) / (
            th.std(features, dim=1, keepdim=True) + 1e-8
        )

        return (
            features,
            th.load(join(self._data_path, f"{file_index}_classes.pt")),
        )

    @property
    def task_type(self) -> networks.TaskType:
        return networks.TaskType.BRAIN_ACTIVITY

    def to_device(self, data: th.Tensor, device: th.device) -> th.Tensor:
        return data.to(device)

    @property
    def delta_t(self) -> float:
        return 1.0
