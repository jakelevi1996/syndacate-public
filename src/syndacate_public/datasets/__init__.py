import juml
from syndacate_public.datasets import (
    objects,
    shapes,
    syndacate,
    generate,
    grapheme,
    words,
    image,
    loader,
    ptpcl,
)

def get_all() -> list[type[juml.base.Dataset]]:
    return [
        *syndacate.get_full_dataset_types(),
        *syndacate.get_small_dataset_types(),
        ptpcl.PreTrainedPartsToClass,
        ptpcl.PreTrainedPartsToClassSmall,
    ]
