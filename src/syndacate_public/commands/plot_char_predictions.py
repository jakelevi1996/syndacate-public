import torch
import numpy as np
from jutility import plotting, cli
import juml
from syndacate_public.datasets import grapheme, generate

class PlotCharPredictions(juml.base.Command):
    @classmethod
    def run(
        cls,
        args:       cli.ParsedArgs,
        n:          int,
        plot_seed:  int,
        threshold:  float,
        plot_type:  juml.tools.PlotType,
    ):
        model_dir, model, dataset = juml.base.Trainer.load(args)

        mp = plotting.MultiPlot(
            cls.plot_split("train", model, dataset, n, plot_seed, threshold),
            cls.plot_split("test",  model, dataset, n, plot_seed, threshold),
            legend=plotting.FigureLegend(
                plotting.Line(c="k", label="Target",        lw=3),
                plotting.Line(c="c", label="Prediction",    lw=9),
                fontsize=20,
            ),
            figsize=[25, 10],
            space=0.1,
        )
        plot_type.plot(mp, "char_predictions", model_dir)

    @classmethod
    def plot_split(
        cls,
        split:      str,
        model:      juml.base.Model,
        dataset:    juml.base.Dataset,
        n:          int,
        plot_seed:  int,
        threshold:  float,
    ) -> plotting.MultiPlot:
        torch.manual_seed(plot_seed)
        data_loader = dataset.get_data_loader(split, n)
        x, t = next(iter(data_loader))
        y = model.forward(x)

        return plotting.MultiPlot(
            *[
                plotting.Subplot(
                    *cls.plot_set(t[i], threshold, 1.5, c="k", z=20),
                    *cls.plot_set(y[i], threshold, 4.5, c="c", z=10),
                    xlim=[-1, 1],
                    ylim=[-1, 1],
                    xticks=[],
                    yticks=[],
                )
                for i in range(n)
            ],
            title="%s Predictions" % split.title(),
        )

    @classmethod
    def plot_set(
        cls,
        pose_set:   torch.Tensor,
        threshold:  float,
        scale:      float,
        **kwargs,
    ) -> list[plotting.Plottable]:
        logits, poses = torch.split(
            pose_set,
            [
                generate.Syndacate.get_num_classes(),
                grapheme.Grapheme.get_pose_dim(),
            ],
            dim=-1,
        )
        t_dim = grapheme.Grapheme.get_pose_names().index("thickness")
        b_dim = grapheme.Grapheme.get_pose_names().index("brightness")
        keep_dims = (poses[:, b_dim] > threshold)
        logits = logits[keep_dims]
        poses = poses[keep_dims]
        poses[:, t_dim] *= scale

        class_inds = logits.argmax(dim=-1, keepdim=False).tolist()
        ind_to_class_name = generate.Syndacate.get_grapheme_names()
        grapheme_names = [
            ind_to_class_name[i]
            for i in class_inds
        ]
        graphemes = [
            grapheme.grapheme_dict[name].from_pose(np.array(p))
            for name, p in zip(grapheme_names, poses.tolist())
        ]
        kwargs.update(show_marker=False)
        return [p for g in graphemes for p in g.plot(**kwargs)]

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("n",            type=int,   default=20),
            cli.Arg("plot_seed",    type=int,   default=0),
            cli.Arg("threshold",    type=float, default=0.1),
            juml.tools.PlotType.get_cli_arg(),
        ]
