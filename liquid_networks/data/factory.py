# -*- coding: utf-8 -*-
from typing import Dict, Final, Literal, Type

from .datasets import (
    AbstractDataset,
    HouseholdPowerDataset,
    MotionSenseDataset,
)

DatasetNames = Literal["household_power", "motion_sense"]

_DATASET_DICT: Final[Dict[str, Type[AbstractDataset]]] = {
    "household_power": HouseholdPowerDataset,
    "motion_sense": MotionSenseDataset,
}


def get_dataset_constructor(dataset_name: str) -> Type[AbstractDataset]:
    return _DATASET_DICT[dataset_name]
