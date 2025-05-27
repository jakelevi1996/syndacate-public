import juml
from syndacate_public.datasets.syndacate.dataset import (
    get_full_dataset_types,
    get_small_dataset_types,
)
from syndacate_public.datasets.ptpcl import (
    PreTrainedPartsToClass,
    PreTrainedPartsToClassSmall,
)

def get_all() -> list[type[juml.base.Dataset]]:
    return [
        *get_full_dataset_types(),
        *get_small_dataset_types(),
        PreTrainedPartsToClass,
        PreTrainedPartsToClassSmall,
    ]
