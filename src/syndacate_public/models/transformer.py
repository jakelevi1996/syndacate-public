import math
import torch
from jutility import cli
import juml

class SelfAttentionLayer(juml.base.Model):
    def __init__(
        self,
        model_dim:  int,
        heads:      int,
    ):
        self._torch_module_init()

        d_qkv = model_dim // heads
        w_scale_qk = 1 / math.sqrt(model_dim * math.sqrt(d_qkv))
        self.unflatten_shape = [heads, d_qkv]

        self.q = juml.models.Linear(model_dim, model_dim, w_scale_qk)
        self.k = juml.models.Linear(model_dim, model_dim, w_scale_qk)
        self.v = juml.models.Linear(model_dim, model_dim)
        self.o = juml.models.Linear(model_dim, model_dim)
        self.rezero_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x_npd: torch.Tensor) -> torch.Tensor:
        q_nhvp = self.q.forward(x_npd).mT.unflatten(-2, self.unflatten_shape)
        k_nhvp = self.k.forward(x_npd).mT.unflatten(-2, self.unflatten_shape)
        v_nhvp = self.v.forward(x_npd).mT.unflatten(-2, self.unflatten_shape)

        g_nhpp = k_nhvp.mT @ q_nhvp
        w_nhpp = torch.softmax(g_nhpp, dim=-2)
        a_nhvp = v_nhvp @ w_nhpp

        a_npd = a_nhvp.flatten(-3, -2).mT
        o_npd = self.o.forward(a_npd)
        x_npd = x_npd + (self.rezero_scale * o_npd)

        return x_npd

class SetTransformer(juml.models.Sequential):
    def __init__(
        self,
        input_shape:    list[int],
        output_shape:   list[int],
        depth:          int,
        model_dim:      int,
        heads:          int,
        expand_ratio:   float,
        embedder:       juml.models.embed.Embedder,
        pooler:         juml.models.pool.Pooler,
    ):
        self._init_sequential(embedder, pooler)
        self.embed.set_input_shape(input_shape)
        self.pool.set_shapes(output_shape, output_shape)

        self.input_linear  = juml.models.Linear(input_shape[-1], model_dim)
        self.layers.append(self.input_linear)

        for _ in range(depth):
            sal = SelfAttentionLayer(model_dim, heads)
            mlp = juml.models.ReZeroMlpLayer(model_dim, expand_ratio)
            self.layers.append(sal)
            self.layers.append(mlp)

        self.output_linear = juml.models.Linear(model_dim, output_shape[-1])
        self.layers.append(self.output_linear)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("depth",        type=int,   default=2),
            cli.Arg("model_dim",    type=int,   default=64),
            cli.Arg("heads",        type=int,   default=8),
            cli.Arg("expand_ratio", type=float, default=2.0, tag="x"),
        ]
