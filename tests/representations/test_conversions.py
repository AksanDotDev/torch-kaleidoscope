import unittest
import torch
import skimage.color as ref_conv
import numpy as np
import hypothesis
import hypothesis.extra.numpy as hnp
import kaleidoscope.representations.conversions as conv


TEST_IMAGE = hnp.arrays(
    np.dtype(np.float32),
    [3, 1, 1],
    elements=hnp.from_dtype(
        np.dtype(np.float32), min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
)


def test_against_reference_conv(image, reference, candidate):
    a = np.ndarray.copy(image)
    b = np.ndarray.copy(image)
    return np.testing.assert_array_almost_equal(
        reference(a, channel_axis=0),
        candidate(torch.from_numpy(b)).numpy()
    )


def test_against_reference_stain(image, reference, candidate):
    a = np.ndarray.copy(image)
    b = np.ndarray.copy(image)
    return np.testing.assert_array_almost_equal(
        ref_conv.separate_stains(a, reference, channel_axis=0),
        candidate(torch.from_numpy(b)).numpy()
    )


class TestConversions(unittest.TestCase):

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_hsv(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2hsv, conv.HSVTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_yuv(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2yuv, conv.YUVTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_yiq(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2yiq, conv.YIQTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_ycbcr(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2ycbcr, conv.YCbCrTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_ypbpr(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2ypbpr, conv.YPbPrTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_ydbdr(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2ydbdr, conv.YDbDrTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_xyz(self, image) -> None:
        self.assertIsNone(test_against_reference_conv(
            image, ref_conv.rgb2xyz, conv.XYZTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_hed(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.hed_from_rgb, conv.HEDTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_hdx(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.hdx_from_rgb, conv.HDXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_fgx(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.fgx_from_rgb, conv.FGXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_bex(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.bex_from_rgb, conv.BEXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_rbd(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.rbd_from_rgb, conv.RBDTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_gdx(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.gdx_from_rgb, conv.GDXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_hax(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.hax_from_rgb, conv.HAXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_bro(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.bro_from_rgb, conv.BROTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_bpx(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.bpx_from_rgb, conv.BPXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_ahx(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.ahx_from_rgb, conv.AHXTransform()
        ))

    @hypothesis.given(image=TEST_IMAGE)
    def test_rgb_to_hpx(self, image) -> None:
        self.assertIsNone(test_against_reference_stain(
            image, ref_conv.hpx_from_rgb, conv.HPXTransform()
        ))


if __name__ == "__main__":
    unittest.main()
