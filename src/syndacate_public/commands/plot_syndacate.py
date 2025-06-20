import torch
import numpy as np
from jutility import plotting, cli
import juml
from syndacate_public.datasets import grapheme, generate, syndacate

class PlotSyndacate(juml.base.Command):
    @classmethod
    def run(
        cls,
        args:       cli.ParsedArgs,
        seed:       int,
        n:          int,
        plot_type:  juml.tools.PlotType,
    ):
        torch.manual_seed(seed)

        sp_list = [
            cls.get_dataset_subplot(d, n)
            for d in [
                syndacate.ImToSingleCharSmall(),
                syndacate.ImToCharsSmall(),
                syndacate.WordsSmall(),
            ]
        ]
        mp = plotting.MultiPlot(
            *sp_list,
            num_cols=1,
            figsize=[8, 4],
            pad=0.04,
        )
        plot_type.plot(mp, "plot_syndacate", "results")

    @classmethod
    def get_dataset_subplot(
        cls,
        d: juml.base.Dataset,
        n: int,
    ) -> plotting.MultiPlot:
        x, t = next(iter(d.get_data_loader("train", n)))
        sp_list = [
            plotting.Subplot(plotting.ImShow(x[i, 0]))
            for i in range(n)
        ]
        return plotting.MultiPlot(*sp_list, num_rows=1)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("seed", type=int,   default=5),
            cli.Arg("n",    type=int,   default=8),
            juml.tools.PlotType.get_cli_arg(),
        ]

    @classmethod
    def include_arg(cls, arg: cli.Arg) -> bool:
        return False
