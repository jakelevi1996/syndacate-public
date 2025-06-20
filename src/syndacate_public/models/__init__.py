from juml.base import Model

from syndacate_public.models.transformer import SetTransformer
from syndacate_public.models.dsts import DeepSetToSet
from syndacate_public.models.capsnet import CapsNet
from syndacate_public.models.mereoformer import MereoFormer

def get_all() -> list[type[Model]]:
    return [
        SetTransformer,
        DeepSetToSet,
        CapsNet,
        MereoFormer,
    ]
