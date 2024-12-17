import pytest

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from pipelines.dependencies.mask_generators.alpha_mask_generator import AlphaMaskGenerator


@pytest.fixture
def alpha_img():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(img)
    drawer.rectangle((4, 4, 6, 6), fill=(255, 0, 0, 255))
    return img

@pytest.fixture
def multiple_alpha_img():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(img)
    drawer.rectangle((2, 2, 4, 4), fill=(255, 0, 0, 255))
    drawer.rectangle((5, 5, 7, 7), fill=(255, 0, 0, 255))
    return img

def test_generate_inside_fill_mask(alpha_img, multiple_alpha_img):
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.inside
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] !=  0:
                assert mask.getpixel((x, y)) == 255
            else:
                assert mask.getpixel((x, y)) == 0


def test_generate_outside_mask(alpha_img, multiple_alpha_img):
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.outside
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] ==  0:
                assert mask.getpixel((x, y)) == 255
            else:
                assert mask.getpixel((x, y)) == 0

def test_generate_border_inside_mask(alpha_img, multiple_alpha_img):
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.outside
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] ==  0:
                assert mask.getpixel((x, y)) == 255
            else:
                assert mask.getpixel((x, y)) == 0

def test_generate_border_border_mask(alpha_img, multiple_alpha_img):
    mask_generator = AlphaMaskGenerator(
        alpha_image=multiple_alpha_img,
        type=AlphaMaskGenerator.Type.outside
    )
    mask = mask_generator.generate()
    for x in range(10):
        for y in range(10):
            if multiple_alpha_img.getpixel((x, y))[3] ==  0:
                assert mask.getpixel((x, y)) == 255
            else:
                assert mask.getpixel((x, y)) == 0
