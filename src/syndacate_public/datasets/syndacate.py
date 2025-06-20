import os
import numpy as np
import torch
import torch.utils.data
from jutility import util, cli
import juml
from syndacate_public.datasets import grapheme, generate, loader

class SampledDataset(juml.datasets.Synthetic):
    def __init__(
        self,
        data_dir:       str=juml.datasets.DATA_REL_DIR,
        force_generate: bool=False,
    ):
        sampler = self.get_sampler(data_dir)
        self.data_path = sampler.get_full_path()
        if (not os.path.isfile(self.data_path)) or force_generate:
            print("Could not find \"%s\", generating now..." % self.data_path)
            sampler.save_dataset()

        x_loader, t_loader = self.get_npz_loaders()
        npz_file = np.load(self.data_path)

        x = x_loader.load(npz_file)
        t = t_loader.load(npz_file)

        split_config        = self.get_split_config()
        split_names         = sorted(split_config.keys())
        split_sizes         = [split_config[k] for k in split_names]
        self._split_dict    = {
            split: juml.datasets.DataSplit(xi, ti, n)
            for split, xi, ti, n in zip(
                split_names,
                torch.split(x, split_sizes, dim=0),
                torch.split(t, split_sizes, dim=0),
                split_sizes,
            )
        }
        self._input_shape   = list(x.shape)
        self._output_shape  = list(t.shape)

    def get_split_config(self) -> dict[str, int]:
        return {"train": 60000, "test": 10000}

    def get_sampler(self, output_dir) -> generate.ImageSampler:
        raise NotImplementedError()

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        raise NotImplementedError()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(cls, tag=cls.get_tag())

    @classmethod
    def get_tag(cls):
        tag = cls.__name__
        replaces_str = (
            "1e:1       2e:2        Single:1    Parts:P     Part:P "
            "Im:I       Chars:C     Char:C      Class:CL    To: "
            "Small:S    Squares:SQ  Multi:      Words:W     Nf:N "
            "Nr:N"
        )
        replaces = [i.split(":") for i in replaces_str.split()]
        for s1, s2 in replaces:
            tag = tag.replace(s1, s2)

        return tag.upper()

    def __len__(self):
        return sum(self.get_split_config().values())

class SingleSquares(SampledDataset):
    def get_sampler(self, output_dir) -> generate.ImageSampler:
        return generate.Squares(
            seed=0,
            num_elements=len(self),
            max_num_objects=1,
            width=100,
            height=100,
            upsample_ratio=10,
            output_dir=output_dir,
        )

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.SingleObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.Mse

class MultiSquares(SampledDataset):
    def get_sampler(self, output_dir) -> generate.ImageSampler:
        return generate.Squares(
            seed=0,
            num_elements=len(self),
            max_num_objects=5,
            width=100,
            height=100,
            upsample_ratio=10,
            output_dir=output_dir,
        )

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.ObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class Words(SampledDataset):
    def get_sampler(self, output_dir) -> generate.ImageSampler:
        return generate.Words(
            seed=0,
            num_elements=len(self),
            max_num_objects=3,
            width=100,
            height=100,
            upsample_ratio=10,
            output_dir=output_dir,
        )

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.ObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class Syndacate(SampledDataset):
    def get_sampler(self, output_dir) -> generate.ImageSampler:
        dataset_kwargs = {
            "seed":                 None,
            "num_elements":         None,
            "max_num_parts":        None,
            "flip_parts":           True,
            "max_num_objects":      1,
            "object_replacement":   True,
            "align_sets":           False,
            "width":                100,
            "height":               100,
            "upsample_ratio":       10,
            "output_dir":           None,
        }
        dataset_kwargs.update(self.get_dataset_config())
        dataset_kwargs["num_elements"] = len(self)
        dataset_kwargs["output_dir"] = output_dir
        sampler = generate.Syndacate(**dataset_kwargs)
        self.class_names = sampler.grapheme_names
        self.num_classes = len(sampler.grapheme_names)
        return sampler

    def get_dataset_config(self) -> dict:
        raise NotImplementedError()

class SyndacateSingleClass(Syndacate):
    def get_output_shape(self) -> list[int]:
        return [generate.Syndacate.get_num_classes()]

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.CrossEntropy

class ImToSinglePart1e(Syndacate):
    def get_dataset_config(self):
        return {"seed": 0, "max_num_parts": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.LineSingleEndpointLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class ImToSinglePart2e(Syndacate):
    def get_dataset_config(self):
        return {"seed": 0, "max_num_parts": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.LineBothEndpointsLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class ImToSinglePart(Syndacate):
    def get_dataset_config(self):
        return {"seed": 0, "max_num_parts": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.PartLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class ImToParts1e(Syndacate):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.LineSingleEndpointLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class ImToParts2e(Syndacate):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.LineBothEndpointsLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class ImToParts(Syndacate):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.PartLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class Parts2eToClassNf(SyndacateSingleClass):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1, "flip_parts": False}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return (
            loader.LineBothEndpointsLoader(),
            loader.GraphemeSingleClassLoader(),
        )

class Parts2eToClass(SyndacateSingleClass):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return (
            loader.LineBothEndpointsLoader(),
            loader.GraphemeSingleClassLoader(),
        )

class PartsToClassNf(SyndacateSingleClass):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1, "flip_parts": False}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.PartLoader(), loader.GraphemeSingleClassLoader()

class PartsToClass(SyndacateSingleClass):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.PartLoader(), loader.GraphemeSingleClassLoader()

class PartsToSingleChar(Syndacate):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.PartLoader(), loader.SingleObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.Mse

class PartsToChars(Syndacate):
    def get_dataset_config(self):
        return {"seed": 3, "max_num_objects": 3, "align_sets": True}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.PartLoader(), loader.ObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.AlignedSetMse

class ImToClass(SyndacateSingleClass):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.GraphemeSingleClassLoader()

class ImToSingleChar(Syndacate):
    def get_dataset_config(self):
        return {"seed": 1, "max_num_objects": 1}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.SingleObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.Mse

class ImToCharsNr(Syndacate):
    def get_dataset_config(self):
        return {"seed": 2, "max_num_objects": 3, "object_replacement": False}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.ObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

class ImToChars(Syndacate):
    def get_dataset_config(self):
        return {"seed": 3, "max_num_objects": 3, "object_replacement": True}

    def get_npz_loaders(self) -> tuple[loader.NpzLoader, loader.NpzLoader]:
        return loader.ImageLoader(), loader.ObjectLoader()

    @classmethod
    def get_default_loss(cls) -> type[juml.base.Loss] | None:
        return juml.loss.ChamferMse

def get_full_dataset_types() -> list[type[SampledDataset]]:
    return [
        SingleSquares,
        MultiSquares,
        ImToSinglePart1e,
        ImToSinglePart2e,
        ImToSinglePart,
        ImToParts1e,
        ImToParts2e,
        ImToParts,
        Parts2eToClassNf,
        Parts2eToClass,
        PartsToClassNf,
        PartsToClass,
        PartsToSingleChar,
        PartsToChars,
        ImToClass,
        ImToSingleChar,
        ImToCharsNr,
        ImToChars,
        Words,
    ]

class Small(SampledDataset):
    def get_split_config(self):
        return {"train": 200, "test": 200}

class SingleSquaresSmall(       SingleSquares,      Small): pass
class MultiSquaresSmall(        MultiSquares,       Small): pass
class ImToSinglePart1eSmall(    ImToSinglePart1e,   Small): pass
class ImToSinglePart2eSmall(    ImToSinglePart2e,   Small): pass
class ImToSinglePartSmall(      ImToSinglePart,     Small): pass
class ImToParts1eSmall(         ImToParts1e,        Small): pass
class ImToParts2eSmall(         ImToParts2e,        Small): pass
class ImToPartsSmall(           ImToParts,          Small): pass
class Parts2eToClassNfSmall(    Parts2eToClassNf,   Small): pass
class Parts2eToClassSmall(      Parts2eToClass,     Small): pass
class PartsToClassNfSmall(      PartsToClassNf,     Small): pass
class PartsToClassSmall(        PartsToClass,       Small): pass
class PartsToSingleCharSmall(   PartsToSingleChar,  Small): pass
class PartsToCharsSmall(        PartsToChars,       Small): pass
class ImToClassSmall(           ImToClass,          Small): pass
class ImToSingleCharSmall(      ImToSingleChar,     Small): pass
class ImToCharsNrSmall(         ImToCharsNr,        Small): pass
class ImToCharsSmall(           ImToChars,          Small): pass
class WordsSmall(               Words,              Small): pass

def get_small_dataset_types() -> list[type[SampledDataset]]:
    return [
        SingleSquaresSmall,
        MultiSquaresSmall,
        ImToSinglePart1eSmall,
        ImToSinglePart2eSmall,
        ImToSinglePartSmall,
        ImToParts1eSmall,
        ImToParts2eSmall,
        ImToPartsSmall,
        Parts2eToClassNfSmall,
        Parts2eToClassSmall,
        PartsToClassNfSmall,
        PartsToClassSmall,
        PartsToSingleCharSmall,
        PartsToCharsSmall,
        ImToClassSmall,
        ImToSingleCharSmall,
        ImToCharsNrSmall,
        ImToCharsSmall,
        WordsSmall,
    ]

def get_all() -> list[type[SampledDataset]]:
    return [
        *get_full_dataset_types(),
        *get_small_dataset_types(),
    ]

dataset_dict = {
    d.__name__: d
    for d in get_all()
}

dataset_names = sorted(dataset_dict.keys())
