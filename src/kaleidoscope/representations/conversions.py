import math as base_math
import torch
from torchvision.transforms.v2 import Grayscale
from typing import Any, Dict

from .representations import ColourRepresentation, register_transform
from ._utilities import ColourRepresentationTransform, MultiplicationTransform, StainSeparationTransform


def fill_third(w):
    w[2, :] = torch.cross(w[0, :], w[1, :], 0)
    return w


identity_matrix = torch.Tensor(
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]]
)

yuv_matrix = torch.Tensor(
    [[0.299, -0.14714119, 0.61497538],
     [0.587, -0.28886916, -0.51496512],
     [0.114, 0.43601035, -0.10001026]]
)

yiq_matrix = torch.Tensor(
    [[0.299, 0.59590059, 0.21153661],
     [0.587, -0.27455667, -0.52273617],
     [0.114, -0.32134392, 0.31119955]]
)

ycbcr_matrix = torch.Tensor(
    [[65.481, -37.797, 112.],
     [128.553, -74.203, -93.786],
     [24.966, 112., -18.214]]
)

ypbpr_matrix = torch.Tensor(
    [[0.299, -0.168736, 0.5],
     [0.587, -0.331264, -0.418688],
     [0.114, 0.5, -0.081312]]
)

ydbdr_matrix = torch.Tensor(
    [[0.299, -0.45, -1.333],
     [0.587, -0.883, 1.116],
     [0.114, 1.333, 0.217]]
)

xyz_matrix = torch.Tensor(
    [[0.412453, 0.212671, 0.019334],
     [0.357580, 0.715160, 0.119193],
     [0.180423, 0.072169, 0.950227]]
)

rgbcie_matrix = torch.Tensor(
    [[0.41846571, -0.09116896, 0.0009209],
     [-0.15866078, 0.25243144, -0.00254981],
     [-0.08283493, 0.01570752, 0.17859891]]
)

hed_matrix = torch.Tensor(
    [[0.65, 0.07, 0.27],
     [0.70, 0.99, 0.57],
     [0.29, 0.11, 0.78]]
).T.inverse()

hdx_matrix = fill_third(torch.Tensor(
    [[0.650, 0.704, 0.286],
     [0.268, 0.570, 0.776],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

fgx_matrix = fill_third(torch.Tensor(
    [[0.46420921, 0.83008335, 0.30827187],
     [0.94705542, 0.25373821, 0.19650764],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

bex_matrix = fill_third(torch.Tensor(
    [[0.834750233, 0.513556283, 0.196330403],
     [0.092789, 0.954111, 0.283111],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

rbd_matrix = torch.Tensor(
    [[0.21393921, 0.85112669, 0.47794022],
     [0.74890292, 0.60624161, 0.26731082],
     [0.268, 0.570, 0.776]]
).T.inverse().transpose(0, 1)

gdx_matrix = fill_third(torch.Tensor(
    [[0.98003, 0.144316, 0.133146],
     [0.268, 0.570, 0.776],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

hax_matrix = fill_third(torch.Tensor(
    [[0.650, 0.704, 0.286],
     [0.2743, 0.6796, 0.6803],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

bro_matrix = torch.Tensor(
    [[0.853033, 0.508733, 0.112656],
     [0.09289875, 0.8662008, 0.49098468],
     [0.10732849, 0.36765403, 0.9237484]]
).T.inverse().transpose(0, 1)

bpx_matrix = fill_third(torch.Tensor(
    [[0.7995107, 0.5913521, 0.10528667],
     [0.09997159, 0.73738605, 0.6680326],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

ahx_matrix = fill_third(torch.Tensor(
    [[0.874622, 0.457711, 0.158256],
     [0.552556, 0.7544, 0.353744],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)

hpx_matrix = fill_third(torch.Tensor(
    [[0.644211, 0.716556, 0.266844],
     [0.175411, 0.972178, 0.154589],
     [0.0, 0.0, 0.0]]
)).T.inverse().transpose(0, 1)


@register_transform
class GrayscaleTransform(ColourRepresentationTransform, Grayscale):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.GRAYSCALE


@register_transform
class RGBTransform(ColourRepresentationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.RGB

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


@register_transform
class HSVTransform(ColourRepresentationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.HSV

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        x_min, v = torch.aminmax(inpt, dim=0)
        c = v - x_min
        h = torch.where(
            # Piecewise for C = 0
            c == 0,
            0.,
            torch.where(
                # Piecewise for V = R
                v == inpt[0],
                (((inpt[1] - inpt[2]) / c) / 6.) % 1.,
                torch.where(
                    # Piecewise for V = G, falling to V = B
                    v == inpt[1],
                    (2. + ((inpt[2] - inpt[0]) / c)) / 6.,
                    (4. + ((inpt[0] - inpt[1]) / c)) / 6.
                )
            )
        )
        s = torch.where(
            v == 0,
            0.,
            c / v
        )
        return torch.stack([h, s, v], dim=0)


@register_transform
class H2SVTransform(HSVTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.H2SV

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        intm = super()._transform(inpt)
        # Scale back to the range 0-2pi and then take the sin and cos
        h_sin = torch.sin(intm[0] * (2 * base_math.pi))
        h_cos = torch.cos(intm[0] * (2 * base_math.pi))
        return torch.stack([h_sin, h_cos, intm[1], intm[2]], dim=0)


@register_transform
class H3SVTransform(HSVTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.H3SV

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        intm = super()._transform(inpt)
        # Scale back to the range 0-2pi and then take the sin and cos
        h_sin = torch.sin(intm[0] * (2 * base_math.pi)) * intm[1]
        h_cos = torch.cos(intm[0] * (2 * base_math.pi)) * intm[1]
        return torch.stack([h_sin, h_cos, intm[1], intm[2]], dim=0)


@register_transform
class YUVTransform(MultiplicationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.YUV
    _transform_matrix = yuv_matrix


@register_transform
class YIQTransform(MultiplicationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.YIQ
    _transform_matrix = yiq_matrix


@register_transform
class YCbCrTransform(MultiplicationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.YCBCR
    _transform_matrix = ycbcr_matrix

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        intm = super()._transform(inpt, dict())
        intm[0, :, :] += 16.0
        intm[1, :, :] += 128.0
        intm[2, :, :] += 128.0
        return intm


@register_transform
class YPbPrTransform(MultiplicationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.YPBPR
    _transform_matrix = ypbpr_matrix


@register_transform
class YDbDrTransform(MultiplicationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.YDBDR
    _transform_matrix = ydbdr_matrix


@register_transform
class XYZTransform(MultiplicationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.XYZ
    _transform_matrix = xyz_matrix

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        intm = torch.where(
            inpt > 0.04045,
            torch.pow((inpt + 0.055) / 1.055, 2.4),
            inpt / 12.92
        )
        intm = super()._transform(intm, dict())
        return intm


@register_transform
class HEDTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.HED
    _transform_matrix = hed_matrix


@register_transform
class HDXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.HDX
    _transform_matrix = hdx_matrix


@register_transform
class FGXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.FGX
    _transform_matrix = fgx_matrix


@register_transform
class BEXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.BEX
    _transform_matrix = bex_matrix


@register_transform
class RBDTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.RBD
    _transform_matrix = rbd_matrix


@register_transform
class GDXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.GDX
    _transform_matrix = gdx_matrix


@register_transform
class HAXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.HAX
    _transform_matrix = hax_matrix


@register_transform
class BROTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.BRO
    _transform_matrix = bro_matrix


@register_transform
class BPXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.BPX
    _transform_matrix = bpx_matrix


@register_transform
class AHXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.AHX
    _transform_matrix = ahx_matrix


@register_transform
class HPXTransform(StainSeparationTransform):
    _from = ColourRepresentation.RGB
    _to = ColourRepresentation.HPX
    _transform_matrix = hpx_matrix
