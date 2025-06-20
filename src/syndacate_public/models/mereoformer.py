import math
import torch
from jutility import cli
import juml

class MereoFormer(juml.models.Sequential):
    def __init__(
        self,
        input_shape:        list[int],
        output_shape:       list[int],
        model_dim:          int,
        heads:              int,
        expand_ratio:       float,
        kernel_size:        int,
        num_stages:         int,
        blocks_per_stage:   int,
        stride:             int,
        embedder:           juml.models.embed.Embedder,
        pooler:             juml.models.pool.Pooler,
    ):
        self._init_sequential(embedder, pooler)
        self.embed.set_input_shape(input_shape)
        self.pool.set_shapes([model_dim, None, None], output_shape)

        for stage in range(num_stages):
            if stage == 0:
                layer = MereoInputLayer(
                    input_channel_dim=self.embed.get_output_dim(-3),
                    model_dim=model_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                self.layers.append(layer)
            else:
                layer = torch.nn.AvgPool3d(
                    kernel_size=[stride, stride, 1],
                    stride=[stride, stride, 1],
                )
                self.layers.append(layer)

            for _ in range(blocks_per_stage):
                mal = MereoAttentionLayer(model_dim, heads, kernel_size)
                mlp = juml.models.ReZeroMlpLayer(model_dim, expand_ratio)
                self.layers.append(mal)
                self.layers.append(mlp)

        layer = MereoOutputLayer()
        self.layers.append(layer)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("model_dim",        type=int,   default=64),
            cli.Arg("heads",            type=int,   default=8),
            cli.Arg("expand_ratio",     type=float, default=2.0, tag="x"),
            cli.Arg("kernel_size",      type=int,   default=5),
            cli.Arg("num_stages",       type=int,   default=3),
            cli.Arg("blocks_per_stage", type=int,   default=2),
            cli.Arg("stride",           type=int,   default=2),
        ]

class MereoInputLayer(juml.base.Model):
    def __init__(
        self,
        input_channel_dim:  int,
        model_dim:          int,
        kernel_size:        int,
        stride:             int,
    ):
        self._torch_module_init()
        self.conv = torch.nn.Conv2d(
            in_channels=input_channel_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x_nihw: torch.Tensor) -> torch.Tensor:
        x_nmhw = self.conv.forward(x_nihw)
        x_nhwm = x_nmhw.transpose(-3, -2).transpose(-2, -1)
        return x_nhwm

class MereoAttentionLayer(juml.base.Model):
    def __init__(
        self,
        model_dim:      int,
        heads:          int,
        kernel_size:    int,
    ):
        self._torch_module_init()

        p1 = (kernel_size // 2)
        p2 = (kernel_size - p1) - 1
        d_qkv = model_dim // heads
        w_scale_qk = 1 / math.sqrt(model_dim * math.sqrt(d_qkv))

        self.pad    = [0, 0, p1, p2, p1, p2]
        self.us_q   = [heads, 1, d_qkv]
        self.us_kv  = [heads, d_qkv]
        self.ks     = kernel_size

        self.q = juml.models.Linear(model_dim, model_dim, w_scale_qk)
        self.k = juml.models.Linear(model_dim, model_dim, w_scale_qk)
        self.v = juml.models.Linear(model_dim, model_dim)
        self.o = juml.models.Linear(model_dim, model_dim)
        self.rezero_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x_nhwm: torch.Tensor) -> torch.Tensor:
        p_nhwm      = torch.nn.functional.pad(x_nhwm, self.pad)
        q_nhwm      = self.q.forward(x_nhwm)
        k_nhwm      = self.k.forward(p_nhwm)
        v_nhwm      = self.v.forward(p_nhwm)

        q_nhwh1v    = q_nhwm.unflatten(-1, self.us_q)
        k_nhwmkk    = k_nhwm.unfold(-3, self.ks, 1).unfold(-3, self.ks, 1)
        v_nhwmkk    = v_nhwm.unfold(-3, self.ks, 1).unfold(-3, self.ks, 1)
        k_nhwhvp    = k_nhwmkk.flatten(-2, -1).unflatten(-2, self.us_kv)
        v_nhwhvp    = v_nhwmkk.flatten(-2, -1).unflatten(-2, self.us_kv)

        g_nhwh1p    = q_nhwh1v @ k_nhwhvp
        w_nhwh1p    = torch.softmax(g_nhwh1p, dim=-1)
        a_nhwhv1    = v_nhwhvp @ w_nhwh1p.mT

        a_nhwm      = a_nhwhv1.flatten(-3, -1)
        o_nhwm      = self.o.forward(a_nhwm)
        x_nhwm      = x_nhwm + (self.rezero_scale * o_nhwm)

        return x_nhwm

class AttentiveDownSampleLayer(juml.base.Model):
    def __init__(
        self,
        model_dim:  int,
        stride:     int,
    ):
        self._torch_module_init()
        self.stride = stride
        self.linear = juml.models.Linear(model_dim, 1)

    def forward(self, x_nhwm: torch.Tensor) -> torch.Tensor:
        s           = self.stride
        g_nhw1      = self.linear.forward(x_nhwm)

        x_nhwmkk    = x_nhwm.unfold(-3, s, s).unfold(-3, s, s)
        g_nhw1kk    = g_nhw1.unfold(-3, s, s).unfold(-3, s, s)
        x_nhwmp     = x_nhwmkk.flatten(-2, -1)
        g_nhwp      = g_nhw1kk.flatten(-3, -1)

        w_nhwp      = torch.softmax(g_nhwp, dim=-1)
        w_nhwp1     = w_nhwp.unsqueeze(-1)
        a_nhwm1     = x_nhwmp @ w_nhwp1
        a_nhwm      = a_nhwm1.squeeze(-1)

        return a_nhwm

class MereoOutputLayer(juml.base.Model):
    def forward(self, x_nhwm: torch.Tensor) -> torch.Tensor:
        x_nmhw = x_nhwm.transpose(-1, -2).transpose(-2, -3)
        return x_nmhw
