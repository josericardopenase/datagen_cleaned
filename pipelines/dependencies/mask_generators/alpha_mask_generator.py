from enum import Enum, IntEnum
from typing import Optional, Callable, Tuple

from PIL import Image
from pydantic import ConfigDict
import numpy as np

from pipelines.dependencies.mask_generators.mask_generators import MaskGenerator


class AlphaMaskGenerator(MaskGenerator):
    class Type(IntEnum):
        inside = 1
        outside = 2
        border = 3

    alpha_image : Image.Image
    type : Type

    border_width : Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def generate(self) -> Image.Image:
        self.alpha_image = self.alpha_image.convert("RGBA")
        if self.type == AlphaMaskGenerator.Type.inside:
                return self.__generate_fill_mask_inside()
        if self.type == AlphaMaskGenerator.Type.outside:
                return self.__generate_fill_mask_outside()
        if self.type == AlphaMaskGenerator.Type.border:
            return self.__generate_border_mask()



    def __map_pixel(self, image : Image.Image, f : Callable[[Image.Image, Tuple[int, int]], None]) -> Image.Image:
        img = image.copy()
        for x in range(self.alpha_image.size[0]):
            for y in range(self.alpha_image.size[1]):
                f(img, (x, y))
        return img

    def __generate_fill_mask_inside(self) -> Image.Image:
        def filter(image : Image.Image, pos : Tuple[int, int]):
            if self.alpha_image.getpixel(pos)[3] != 0:
                image.putpixel(pos, 255)
        return self.__map_pixel(Image.new("L", self.alpha_image.size, 0), filter)

    def __generate_fill_mask_outside(self) -> Image.Image:
        def filter(image : Image.Image, pos : Tuple[int, int]):
            if self.alpha_image.getpixel(pos)[3] == 0:
                image.putpixel(pos, 255)
        return self.__map_pixel(Image.new("L", self.alpha_image.size), filter)

    def __generate_border_mask(self) -> Image.Image:
        mask = Image.new("L", self.alpha_image.size, 0)
        return mask
