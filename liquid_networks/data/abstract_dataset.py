from abc import ABC, abstractmethod

import torch as th
from torch.utils.data import Dataset

from liquid_networks import networks


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
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        pass

    @property
    @abstractmethod
    def task_type(self) -> networks.TaskType:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(data_path={self._data_path})"

    def __repr__(self) -> str:
        return str(self)
