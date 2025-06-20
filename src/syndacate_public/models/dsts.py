import torch
from jutility import cli
import juml

class DeepSetToSet(juml.models.Model):
    def __init__(
        self,
        input_shape:    list[int],
        output_shape:   list[int],
        hidden_dim:     int,
        depth:          int,
        embedder:       juml.models.embed.Embedder,
        pooler:         juml.models.pool.Pooler,
    ):
        self._torch_module_init()
        self.embed  = embedder
        self.pool   = pooler
        self.embed.set_input_shape(input_shape)
        self.pool.set_shapes(output_shape, output_shape)

        self.linear = juml.models.Linear(
            input_dim=(self.embed.get_output_dim(-1)),
            output_dim=(hidden_dim),
        )
        self.mlp_1 = juml.models.Mlp(
            input_shape=[self.embed.get_output_dim(-1)],
            output_shape=[hidden_dim],
            hidden_dim=hidden_dim,
            depth=depth,
            embedder=juml.models.embed.Identity(),
            pooler=juml.models.pool.Identity(),
        )
        self.mlp_2 = juml.models.Mlp(
            input_shape=[hidden_dim],
            output_shape=[self.pool.get_input_dim(-1)],
            hidden_dim=hidden_dim,
            depth=depth,
            embedder=juml.models.embed.Identity(),
            pooler=juml.models.pool.Identity(),
        )

    def forward(self, x_npi: torch.Tensor):
        x_npi   = self.embed.forward(x_npi)
        x1_nph  = self.mlp_1.forward(x_npi)
        x1_n1h  = x1_nph.sum(dim=-2, keepdim=True)
        x2_nph  = self.linear.forward(x_npi)
        x_nph   = x1_n1h + x2_nph
        x_npo   = self.mlp_2.forward(x_nph)
        x_npo   = self.pool.forward(x_npo)
        return x_npo

    @classmethod
    def get_cli_options(cls):
        return [
            cli.Arg("hidden_dim",   type=int, default=100),
            cli.Arg("depth",        type=int, default=3),
            juml.models.embed.get_cli_choice(),
            juml.models.pool.get_cli_choice(),
        ]
