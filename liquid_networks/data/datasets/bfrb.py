import re
from os import listdir
from os.path import join
from typing import Any, Callable

import torch as th
from torch.nn import functional as th_f

from liquid_networks import networks

from ..abstract_dataset import AbstractDataset, AbstractDatasetFactory


class BfrbDataset(AbstractDataset[tuple[th.Tensor, th.Tensor]]):
    def __init__(self, data_path: str, normalize_grid: bool, normalize_features: bool) -> None:
        super().__init__(data_path)

        regex_target = re.compile(r"^(.+)_target\.pth$")

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
            min_value = 0.0
            max_value = 255.0
            grids = grids / (max_value - min_value) * 2.0 - 1.0

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
                [
                    th_f.pad(
                        x, (0, 0, 0, 0, 0, 0, max_len - x.size(0), 0), mode="constant", value=0.0
                    )
                    for x in grids
                ],
                dim=0,
            )

            features_tensor = th.stack(
                [
                    th_f.pad(x, (0, 0, max_len - x.size(0), 0), "constant", value=0.0)
                    for x in features
                ],
                dim=0,
            )

            return (grids_tensor, features_tensor), th.cat(targets, dim=0)

        return __collate

    def to_device(
        self, data: tuple[th.Tensor, th.Tensor], device: th.device
    ) -> tuple[th.Tensor, th.Tensor]:
        return data[0].to(device), data[1].to(device)

    @property
    def delta_t(self) -> float:
        return 1.0


class BfrbDatasetFactory(AbstractDatasetFactory[tuple[th.Tensor, th.Tensor]]):
    def get_dataset(self, data_path: str) -> AbstractDataset[tuple[th.Tensor, th.Tensor]]:
        return BfrbDataset(
            data_path,
            self._get_config("normalize_grid", bool, True),
            self._get_config("normalize_features", bool, True),
        )
