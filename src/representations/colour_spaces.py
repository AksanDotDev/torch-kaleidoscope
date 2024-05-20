import math as base_math
import torch
import torch.nn as nn

__all__ = []  # TODO Double bind functions and fill this out


def fill_third(w):
    w[2, :] = torch.cross(w[0, :], w[1, :], 0)
    return w


identity_weights = nn.Parameter(torch.Tensor(
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]]
).reshape([3, 3, 1, 1]))

xyz_weights = nn.Parameter(torch.Tensor(
    [[0.49, 0.17697, 0.],
     [0.31, 0.8124, 0.01],
     [0.2, 0.01063, 0.99]]
).reshape([3, 3, 1, 1]))

yuv_weights = nn.Parameter(torch.Tensor(
    [[0.299, -0.14714119, 0.61497538],
     [0.587, -0.28886916, -0.51496512],
     [0.114, 0.43601035, -0.10001026]]
).reshape([3, 3, 1, 1]))

yiq_weights = nn.Parameter(torch.Tensor(
    [[0.299, 0.59590059, 0.21153661],
     [0.587, -0.27455667, -0.52273617],
     [0.114, -0.32134392, 0.31119955]]
).reshape([3, 3, 1, 1]))

ycbcr_weights = nn.Parameter(torch.Tensor(
    [[65.481, -37.797, 112.],
     [128.553, -74.203, -93.786],
     [24.966, 112., -18.214]]
).reshape([3, 3, 1, 1]))

ypbpr_weights = nn.Parameter(torch.Tensor(
    [[0.299, -0.168736, 0.5],
     [0.587, -0.331264, -0.418688],
     [0.114, 0.5, -0.081312]]
).reshape([3, 3, 1, 1]))

ydbdr_weights = nn.Parameter(torch.Tensor(
    [[0.299, -0.45, -1.333],
     [0.587, -0.883, 1.116],
     [0.114, 1.333, 0.217]]
).reshape([3, 3, 1, 1]))

rgbcie_weights = nn.Parameter(torch.Tensor(
    [[0.41846571, -0.09116896, 0.0009209],
     [-0.15866078, 0.25243144, -0.00254981],
     [-0.08283493, 0.01570752, 0.17859891]]
).reshape([3, 3, 1, 1]))

hed_weights = nn.Parameter(torch.Tensor(
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]]
).T.inverse().reshape([3, 3, 1, 1]))

hdx_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.650, 0.704, 0.286],
     [0.268, 0.570, 0.776],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

fgx_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.46420921, 0.83008335, 0.30827187],
     [0.94705542, 0.25373821, 0.19650764],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

bex_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.834750233, 0.513556283, 0.196330403],
     [0.092789, 0.954111, 0.283111],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

rbd_weights = nn.Parameter(torch.Tensor(
    [[0.21393921, 0.85112669, 0.47794022],
     [0.74890292, 0.60624161, 0.26731082],
     [0.268, 0.570, 0.776]]
).T.inverse().reshape([3, 3, 1, 1]))

gdx_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.98003, 0.144316, 0.133146],
     [0.268, 0.570, 0.776],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

hax_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.650, 0.704, 0.286],
     [0.2743, 0.6796, 0.6803],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

bro_weights = nn.Parameter(torch.Tensor(
    [[0.853033, 0.508733, 0.112656],
     [0.09289875, 0.8662008, 0.49098468],
     [0.10732849, 0.36765403, 0.9237484]]
).T.inverse().reshape([3, 3, 1, 1]))

bpx_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.7995107, 0.5913521, 0.10528667],
     [0.09997159, 0.73738605, 0.6680326],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

ahx_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.874622, 0.457711, 0.158256],
     [0.552556, 0.7544, 0.353744],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))

hpx_weights = nn.Parameter(fill_third(torch.Tensor(
    [[0.644211, 0.716556, 0.266844],
     [0.175411, 0.972178, 0.154589],
     [0.0, 0.0, 0.0]]
)).T.inverse().reshape([3, 3, 1, 1]))


class Fixed1DConv(nn.Conv2d):
    input_channels = 3

    def __init__(self, weights=None):
        super().__init__(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="zeros",
            dilation=1,
            groups=1,
            bias=False
        )
        if weights is not None:
            self.fix_weights(weights)

    def fix_weights(self, weights):
        self.weight = weights
        self.weight.requires_grad = False


class FixedIDConv(Fixed1DConv):
    def __init__(self):
        super().__init__(identity_weights)


class FixedXYZConv(Fixed1DConv):
    def __init__(self):
        super().__init__(xyz_weights)


class FixedYUVConv(Fixed1DConv):
    def __init__(self):
        super().__init__(yuv_weights)


class FixedYIQConv(Fixed1DConv):
    def __init__(self):
        super().__init__(yiq_weights)


class FixedYCbCrConv(Fixed1DConv):
    def __init__(self):
        super().__init__(ycbcr_weights)


class FixedYPbPrConv(Fixed1DConv):
    def __init__(self):
        super().__init__(ypbpr_weights)


class FixedYDbDrConv(Fixed1DConv):
    def __init__(self):
        super().__init__(ydbdr_weights)


class FixedRGBCIEConv(Fixed1DConv):
    def __init__(self):
        super().__init__(rgbcie_weights)


class HSVLayer(nn.Module):
    input_channels = 3

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x_min, v = torch.aminmax(x, dim=0)
        c = v - x_min
        h = torch.where(
            # Piecewise for C = 0
            c == 0,
            0.,
            torch.where(
                # Piecewise for V = R
                v == x[0],
                (((x[1] - x[2]) / c) / 6.) % 1.,
                torch.where(
                    # Piecewise for V = G, falling to V = B
                    v == x[1],
                    (2. + ((x[2] - x[0]) / c)) / 6.,
                    (4. + ((x[0] - x[1]) / c)) / 6.
                )
            )
        )
        s = torch.where(
            v == 0,
            0.,
            c / v
        )
        return torch.stack([h, s, v], dim=0)


class H2SVLayer(HSVLayer):
    input_channels = 4

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = super().forward(x)
        # Scale back to the range 0-2pi and then take the sin and cos
        h_sin = torch.sin(x[0] * (2 * base_math.pi))
        h_cos = torch.cos(x[0] * (2 * base_math.pi))
        return torch.stack([h_sin, h_cos, x[1], x[2]], dim=0)


class H3SVLayer(HSVLayer):
    input_channels = 4

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = super().forward(x)
        # Scale back to the range 0-2pi and then take the sin and cos
        h_sin = torch.sin(x[0] * (2 * base_math.pi)) * x[1]
        h_cos = torch.cos(x[0] * (2 * base_math.pi)) * x[1]
        return torch.stack([h_sin, h_cos, x[1], x[2]], dim=0)


class FixedLog1DConv(Fixed1DConv):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self.register_buffer(
            "_log_artifact",
            torch.tensor(1E-6),
            False
        )
        self.register_buffer(
            "_log_adjust",
            torch.tensor(1E-6).log(),
            False
        )
        self.register_buffer(
            "_zero",
            torch.tensor(0.0),
            False
        )

    def forward(self, x):
        x = torch.max(x, self._log_artifact)

        x = super().forward(x.log() / self._log_adjust)

        x = torch.max(x, self._zero)

        return x


class FixedLogHEDConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(hed_weights)


class FixedLogHDXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(hdx_weights)


class FixedLogFGXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(fgx_weights)


class FixedLogBEXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(bex_weights)


class FixedLogRBDConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(rbd_weights)


class FixedLogGDXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(gdx_weights)


class FixedLogHAXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(hax_weights)


class FixedLogBROConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(bro_weights)


class FixedLogBPXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(bpx_weights)


class FixedLogAHXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(ahx_weights)


class FixedLogHPXConv(FixedLog1DConv):
    def __init__(self):
        super().__init__(hpx_weights)
