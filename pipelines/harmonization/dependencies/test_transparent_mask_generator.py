from PIL import Image
from matplotlib import pyplot as plt
from mypy.types import TupleType
from pipelines.harmonization.dependencies.transparent_mask_generator import TransparentMaskGenerator

from PIL import Image, ImageDraw
from typing import Tuple

class BoxMaskGenerator:
    def generate(self, resolution: Tuple[int, int], mask_size: Tuple[int, int]) -> Image.Image:
        img = Image.new('L', resolution, 0)
        draw = ImageDraw.Draw(img)
        x_start = (resolution[0] - mask_size[0]) / 2
        y_start = (resolution[1] - mask_size[1]) / 2
        x_end = x_start + mask_size[0]
        y_end = y_start + mask_size[1]
        draw.rectangle([x_start, y_start, x_end, y_end], fill=255)
        return img

def test_transparent_mask_generator():
    mask_generator = BoxMaskGenerator()
    fg = Image.open("assets/boat.png")
    plt.imshow(mask_generator.generate(fg.size, (300, 300)))
    plt.show()