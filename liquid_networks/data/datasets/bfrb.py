import json
import re
from os import listdir
from os.path import join
from typing import Any, Callable

import pandas as pd
import torch as th
from torch import Tensor
from torch.nn import functional as th_f

from liquid_networks import networks

from ..abstract_dataset import AbstractDataset, AbstractDatasetFactory
from ..prediction_register import AbstractPredictionRegister, NoPredictionRegister

# With Grids


class BfrbDataset(AbstractDataset[tuple[th.Tensor, th.Tensor]]):
    def __init__(self, data_path: str, normalize_grid: bool, normalize_features: bool) -> None:
        super().__init__(data_path)

        regex_target = re.compile(r"^(.+)_features\.pth$")

        all_files = listdir(self._data_path)
        sequence_id = []
        for f in all_files:
            match = regex_target.match(f)
            if match is not None:
                sequence_id.append(match.group(1))

        self.__idx_to_sequence_id = dict(enumerate(sequence_id))
        self.__normalize_grid = normalize_grid
        self.__normalize_features = normalize_features

    def __len__(self) -> int:
        return len(self.__idx_to_sequence_id)

    def __getitem__(self, index: int) -> tuple[tuple[th.Tensor, th.Tensor], th.Tensor]:
        target = th.load(join(self._data_path, f"{self.__idx_to_sequence_id[index]}_target.pth"))

        grids = th.load(join(self._data_path, f"{self.__idx_to_sequence_id[index]}_grids.pth")).to(
            th.float
        )

        if self.__normalize_grid:
            grids = (grids + 1.0) / 256.0 * 2.0 - 1.0

        features = th.load(
            join(self._data_path, f"{self.__idx_to_sequence_id[index]}_features.pth")
        )

        if self.__normalize_features:
            # std normalized over time (dim=0)
            features = (features - features.mean(dim=0, keepdim=True)) / (
                features.std(dim=0, keepdim=True) + 1e-8
            )

        return (grids, features), target

    @property
    def task_type(self) -> networks.TaskType:
        return networks.TaskType.BFRB

    @property
    def collate_fn(self) -> Callable[[list], Any] | None:
        def __collate(
            batch: list[tuple[tuple[th.Tensor, th.Tensor], th.Tensor]],
        ) -> tuple[tuple[th.Tensor, th.Tensor], th.Tensor]:
            grids_features, targets = zip(*batch)
            grids, features = zip(*grids_features)

            max_len = max(map(lambda g: g.size(0), grids))

            grids_tensor = th.stack(
                [th_f.pad(x, (*[0] * 6, max_len - x.size(0), 0)) for x in grids],
                dim=0,
            )

            features_tensor = th.stack(
                [th_f.pad(x, (0, 0, max_len - x.size(0), 0)) for x in features],
                dim=0,
            )

            return (grids_tensor, features_tensor), th.cat(targets, dim=0)

        return __collate

    def to_device(
        self, data: tuple[th.Tensor, th.Tensor], device: th.device
    ) -> tuple[th.Tensor, th.Tensor]:
        return data[0].to(device), data[1].to(device)

    def get_prediction_register(self) -> AbstractPredictionRegister:
        return NoPredictionRegister()


class BfrbDatasetFactory(AbstractDatasetFactory[tuple[th.Tensor, th.Tensor]]):
    def get_dataset(self, data_path: str) -> AbstractDataset[tuple[th.Tensor, th.Tensor]]:
        return BfrbDataset(
            data_path,
            self._get_config("normalize_grid", bool, True),
            self._get_config("normalize_features", bool, True),
        )


# Without grids


class BfrbFeauresOnlyPredictionRegister(AbstractPredictionRegister):
    def __init__(self, get_data_id_fn: Callable[[int], str], idx_to_class: dict[int, str]) -> None:
        super().__init__(get_data_id_fn)

        self.__idx_to_class = idx_to_class

        self.__data: list[tuple[str, str]] = []

    def _register_impl(self, data_id: str, prediction: Tensor) -> None:
        self.__data.append((data_id, self.__idx_to_class[int(th.argmax(prediction).item())]))

    def to_file(self, output_folder: str) -> None:
        pd.DataFrame(self.__data, columns=["sequence_id", "gesture"]).to_csv(
            join(output_folder, "sample_submission.csv"), sep=",", index=False
        )


class BfrbFeaturesOnlyDataset(AbstractDataset[th.Tensor]):
    def __init__(self, data_path: str, normalize_features: bool) -> None:
        super().__init__(data_path)

        regex_target = re.compile(r"^(.+)_features\.pth$")

        all_files = listdir(self._data_path)
        sequence_id = []
        for f in all_files:
            match = regex_target.match(f)
            if match is not None:
                sequence_id.append(match.group(1))

        self.__idx_to_sequence_id = dict(enumerate(sequence_id))
        self.__normalize_features = normalize_features

        with open(join(self._data_path, "class_to_idx.json"), "r", encoding="utf-8") as json_file:
            self.__idx_to_class = {
                class_idx: class_name for class_name, class_idx in json.load(json_file).items()
            }

    def __len__(self) -> int:
        return len(self.__idx_to_sequence_id)

    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]:
        target = th.load(join(self._data_path, f"{self.__idx_to_sequence_id[index]}_target.pth"))

        features = th.load(
            join(self._data_path, f"{self.__idx_to_sequence_id[index]}_features.pth")
        )

        if self.__normalize_features:
            # std normalized over time (dim=0)
            features = (features - features.mean(dim=0, keepdim=True)) / (
                features.std(dim=0, keepdim=True) + 1e-8
            )

        return features, target

    @property
    def task_type(self) -> networks.TaskType:
        return networks.TaskType.BFRB_FEATURES

    @property
    def collate_fn(self) -> Callable[[list], Any] | None:
        def __collate(
            batch: list[tuple[th.Tensor, th.Tensor]],
        ) -> tuple[th.Tensor, th.Tensor]:
            features, targets = zip(*batch)

            max_len = max(map(lambda f: f.size(0), features))

            features_tensor = th.stack(
                [th_f.pad(x, (0, 0, max_len - x.size(0), 0)) for x in features],
                dim=0,
            )

            return features_tensor, th.cat(targets, dim=0)

        return __collate

    def to_device(self, data: th.Tensor, device: th.device) -> th.Tensor:
        return data.to(device)

    def get_prediction_register(self) -> AbstractPredictionRegister:
        return BfrbFeauresOnlyPredictionRegister(
            lambda idx: self.__idx_to_sequence_id[idx],
            self.__idx_to_class,
        )


class BfrbFeaturesOnlyDatasetFactory(AbstractDatasetFactory[th.Tensor]):
    def get_dataset(self, data_path: str) -> AbstractDataset[th.Tensor]:
        return BfrbFeaturesOnlyDataset(
            data_path,
            self._get_config("normalize_features", bool, True),
        )
