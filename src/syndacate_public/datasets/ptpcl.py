import os
import numpy as np
import torch
from jutility import util, cli
import juml
from syndacate_public.datasets.syndacate import (
    ImToClass,
    ImToClassSmall,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

class PreTrainedPartsToClass(juml.datasets.DatasetFromDict):
    def __init__(
        self,
        data_dir:       str=juml.datasets.DATA_REL_DIR,
        force_generate: bool=False,
    ):
        self.data_path = util.get_full_path(
            "%s.npz" % type(self).__name__,
            juml.datasets.DATA_REL_DIR,
            loading=True,
        )
        if (not os.path.isfile(self.data_path)) or force_generate:
            print("Could not find \"%s\", generating now..." % self.data_path)
            self.save_dataset()

        npz_file = np.load(self.data_path)
        self._split_dict = {
            "train": juml.datasets.DataSplit(
                x=torch.tensor(npz_file["x_train"], dtype=torch.float32),
                t=torch.tensor(npz_file["t_train"], dtype=torch.int64),
                n=npz_file["n_train"],
            ),
            "test": juml.datasets.DataSplit(
                x=torch.tensor(npz_file["x_test"], dtype=torch.float32),
                t=torch.tensor(npz_file["t_test"], dtype=torch.int64),
                n=npz_file["n_test"],
            ),
        }

    def save_dataset(self):
        model = juml.models.RzCnn(
            input_shape=[70000, 1, 100, 100],
            output_shape=[70000, 9, 6],
            kernel_size=5,
            model_dim=64,
            expand_ratio=2.0,
            num_stages=3,
            blocks_per_stage=2,
            stride=2,
            embedder=juml.models.embed.CoordConv(),
            pooler=juml.models.pool.LinearSet2d(),
        )

        model_name = (
            "dIP_lCH_mRZCb2eCk5m64n3pLs2x2.0_"
            "tBb100e100lCle1E-05oAol0.001_s3"
        )
        model_dir = os.path.join(ROOT_DIR, "results", "train", model_name)
        model_path = util.get_full_path("model.pth", model_dir, loading=True)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        self.encoder, _ = model.split(len(model.layers))
        self.encoder.requires_grad_(False)

        self.dataset = self.get_pretrain_dataset()
        split_config = self.dataset.get_split_config()

        y_train, t_train = self.eval_split("train")
        y_test,  t_test  = self.eval_split("test" )

        np.savez_compressed(
            self.data_path,
            x_train=y_train.numpy(),
            t_train=t_train.numpy(),
            n_train=split_config["train"],
            x_test=y_test.numpy(),
            t_test=t_test.numpy(),
            n_test=split_config["test"],
        )

    def get_pretrain_dataset(self) -> ImToClass:
        return ImToClass()

    def eval_split(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        train_loader = self.dataset.get_data_loader(split, 100)
        y_list = []
        t_list = []
        for x, t in util.progress(train_loader):
            y = self.encoder.forward(x)
            y_list.append(y)
            t_list.append(t)

        y_full = torch.concatenate(y_list, dim=0)
        t_full = torch.concatenate(t_list, dim=0)

        return y_full, t_full

    def get_input_shape(self) -> list[int]:
        return [64, 9, 9]

    def get_output_shape(self) -> list[int]:
        return [10]

    def get_default_loss(self) -> type[juml.base.Loss] | None:
        return juml.loss.CrossEntropy

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(cls, tag="PTPCL")

class PreTrainedPartsToClassSmall(PreTrainedPartsToClass):
    def get_pretrain_dataset(self) -> ImToClass:
        return ImToClassSmall()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(cls, tag="PTPCLS")
