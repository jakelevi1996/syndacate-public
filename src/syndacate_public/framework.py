import juml
import syndacate_public as sn

class Framework(juml.base.Framework):
    @classmethod
    def get_models(cls) -> list[type[juml.base.Model]]:
        return [
            *juml.models.get_all(),
            *sn.models.get_all(),
        ]

    @classmethod
    def get_datasets(cls) -> list[type[juml.base.Dataset]]:
        return [
            *sn.datasets.get_all(),
        ]

    @classmethod
    def get_commands(cls) -> list[type[juml.base.Command]]:
        return [
            *juml.commands.get_all(),
            *sn.commands.get_all(),
        ]
