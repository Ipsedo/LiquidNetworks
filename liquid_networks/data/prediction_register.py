from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor


class AbstractPredictionRegister(ABC):
    def __init__(self, get_data_id_fn: Callable[[int], str]) -> None:
        self.__get_data_id = get_data_id_fn

    @abstractmethod
    def _register_impl(self, data_id: str, prediction: Tensor) -> None:
        pass

    def register(self, data_idx: int, prediction: Tensor) -> None:
        self._register_impl(self.__get_data_id(data_idx), prediction)

    def register_batch(self, start_data_idx: int, prediction: Tensor) -> None:
        assert len(prediction.size()) >= 1

        for data_idx in range(start_data_idx, start_data_idx + prediction.size(0)):
            self.register(data_idx, prediction[data_idx - start_data_idx])

    @abstractmethod
    def to_file(self, output_folder: str) -> None:
        pass


class NoPredictionRegister(AbstractPredictionRegister):

    def __init__(self) -> None:
        super().__init__(str)

    def _register_impl(self, data_id: str, prediction: Tensor) -> None:
        pass

    def to_file(self, output_folder: str) -> None:
        pass
