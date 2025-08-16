from typing import Final, Literal, Type

from .datasets import (
    AbstractDataset,
    HarmfulBrainActivityDataset,
    HouseholdPowerDataset,
    MotionSenseDataset,
)

DatasetNames = Literal["household_power", "motion_sense", "brain_activity"]

_DATASET_DICT: Final[dict[str, Type[AbstractDataset]]] = {
    "household_power": HouseholdPowerDataset,
    "motion_sense": MotionSenseDataset,
    "brain_activity": HarmfulBrainActivityDataset,
}


def get_dataset_constructor(
    dataset_name: DatasetNames,
) -> Type[AbstractDataset]:
    return _DATASET_DICT[dataset_name]
