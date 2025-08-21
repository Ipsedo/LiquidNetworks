from typing import Final, Literal, Type

from .abstract_dataset import AbstractDataset
from .datasets import (
    BfrbDataset,
    HarmfulBrainActivityDataset,
    HouseholdPowerDataset,
    MotionSenseDataset,
)

DatasetNames = Literal[
    "household_power", "motion_sense", "brain_activity", "bfrb"
]

_DATASET_DICT: Final[dict[str, Type[AbstractDataset]]] = {
    "household_power": HouseholdPowerDataset,
    "motion_sense": MotionSenseDataset,
    "brain_activity": HarmfulBrainActivityDataset,
    "bfrb": BfrbDataset,
}


def get_dataset_constructor(
    dataset_name: DatasetNames,
) -> Type[AbstractDataset]:
    return _DATASET_DICT[dataset_name]
