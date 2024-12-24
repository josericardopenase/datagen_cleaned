from typing import Tuple

from PIL import Image
from matplotlib import pyplot as plt

from core.dependencies.ai.discriminative_ai.point_extractors.mock_point_extractor import MockPointExtractor
from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor
from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from core.dependencies.ai.generative_ai.image_generators.mock_image_generator import MockImageGenerator
from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter
from core.dependencies.ai.generative_ai.image_inpainters.mock_image_inpainter import MockImageInpainter
from core.dependencies.utils.image.image_editors.image_paster import ImagePaster
from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator
from core.pipelines.pipeline import DatasetGenerationPipeline, BoundingBox


class OutpaintingDatasetGenerator(DatasetGenerationPipeline):
    fg_generator : ImageGenerator
    point_extractor : PointExtractor
    inpainter : ImageInpainter

    def generate(self) -> Tuple[Image.Image, BoundingBox]:
        fg = self.fg_generator.generate()
        fg = fg.resize((300, ((fg.size[1]*300)//fg.size[0])))
        bg = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        point = self.point_extractor.extract(bg)
        bg_with_boat = ImagePaster(center=point, patch=fg).edit(bg)
        inpainting_mask = AlphaMaskGenerator(alpha_image=bg_with_boat, type=AlphaMaskGenerator.Type.outside).generate()
        bg_inpainted = self.inpainter.inpaint(bg_with_boat, inpainting_mask, prompt="A boat with water trails")
        return bg_inpainted, BoundingBox(x=0, y=0, w=0, h=0)

if __name__ == "__main__":
    img, bbox = OutpaintingDatasetGenerator(
        fg_generator=MockImageGenerator(route="../../../data_assets/boats/without_bg/image_1-removebg-preview.png"),
        point_extractor=MockPointExtractor(point=(250, 250)),
        inpainter=MockImageInpainter()
    ).generate()
    img.save("outpainting.png")
