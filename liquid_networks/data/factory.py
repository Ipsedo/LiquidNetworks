from enum import StrEnum
from typing import Final, Type

from .abstract_dataset import AbstractDataset
from .datasets import (
    BfrbDataset,
    HarmfulBrainActivityDataset,
    HouseholdPowerDataset,
    MotionSenseDataset,
)


class DatasetNames(StrEnum):
    HOUSEHOLD_POWER = "household_power"
    MOTION_SENSE = "motion_sense"
    BRAIN_ACTIVITY = "brain_activity"
    BFRB = "bfrb"


_DATASET_DICT: Final[dict[DatasetNames, Type[AbstractDataset]]] = {
    DatasetNames.HOUSEHOLD_POWER: HouseholdPowerDataset,
    DatasetNames.MOTION_SENSE: MotionSenseDataset,
    DatasetNames.BRAIN_ACTIVITY: HarmfulBrainActivityDataset,
    DatasetNames.BFRB: BfrbDataset,
}


def get_dataset_constructor(
    dataset_name: DatasetNames,
) -> Type[AbstractDataset]:
    return _DATASET_DICT[dataset_name]
