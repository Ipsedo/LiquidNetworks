from enum import StrEnum
from typing import Final, Type

from .abstract_dataset import AbstractDataset, AbstractDatasetFactory
from .datasets import (
    BfrbDatasetFactory,
    BfrbFeaturesOnlyDatasetFactory,
    HarmfulBrainActivityDatasetFactory,
    HouseholdPowerDatasetFactory,
    MotionSenseDatasetFactory,
)


class DatasetNames(StrEnum):
    HOUSEHOLD_POWER = "household_power"
    MOTION_SENSE = "motion_sense"
    BRAIN_ACTIVITY = "brain_activity"
    BFRB = "bfrb"
    BFRB_FEATURES = "bfrb_features"


_DATASET_DICT: Final[dict[DatasetNames, Type[AbstractDatasetFactory]]] = {
    DatasetNames.HOUSEHOLD_POWER: HouseholdPowerDatasetFactory,
    DatasetNames.MOTION_SENSE: MotionSenseDatasetFactory,
    DatasetNames.BRAIN_ACTIVITY: HarmfulBrainActivityDatasetFactory,
    DatasetNames.BFRB: BfrbDatasetFactory,
    DatasetNames.BFRB_FEATURES: BfrbFeaturesOnlyDatasetFactory,
}


def get_dataset_constructor(dataset_name: DatasetNames) -> Type[AbstractDatasetFactory]:
    return _DATASET_DICT[dataset_name]
