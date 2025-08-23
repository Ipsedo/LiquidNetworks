from abc import ABC, abstractmethod
from typing import Any, Callable

import torch as th
from torch.utils.data import Dataset

from liquid_networks import networks
from liquid_networks.factory import BaseFactory


class AbstractDataset[T](ABC, Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()

        self._data_path = data_path

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[T, th.Tensor]:
        pass

    @property
    @abstractmethod
    def delta_t(self) -> float:
        pass

    @property
    @abstractmethod
    def task_type(self) -> networks.TaskType:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(data_path={self._data_path})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def collate_fn(self) -> Callable[[list], Any] | None:
        return None

    @abstractmethod
    def to_device(self, data: T, device: th.device) -> T:
        pass


class AbstractDatasetFactory[T](BaseFactory, ABC):
    @abstractmethod
    def get_dataset(self, data_path: str) -> AbstractDataset[T]:
        pass
