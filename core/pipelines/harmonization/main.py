from typing import Tuple
from PIL import Image
from matplotlib import pyplot as plt

from core.dependencies.ai.discriminative_ai.background_removers.background_remover import BackgroundRemover
from core.dependencies.ai.discriminative_ai.background_removers.mock_background_remover import MockBackgroundRemover
from core.dependencies.ai.discriminative_ai.point_extractors.mock_point_extractor import MockPointExtractor
from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor
from core.dependencies.ai.generative_ai.image_generators.mock_image_generator import MockImageGenerator
from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator
from core.dependencies.ai.generative_ai.image_harmonizers.image_harmonizer import ImageHarmonizer
from core.dependencies.ai.generative_ai.image_harmonizers.mock_image_harnonizer import MockImageHarmonizer
from core.dependencies.ai.generative_ai.image_inpainters.image_inpainter import ImageInpainter
from core.dependencies.ai.generative_ai.image_inpainters.mock_image_inpainter import MockImageInpainter
from core.dependencies.utils.image.image_editors.image_cropper import ImageCropper
from core.dependencies.utils.image.image_editors.image_paster import ImagePaster
from core.dependencies.utils.image.mask_generators.alpha_mask_generator import AlphaMaskGenerator
from core.dependencies.utils.image.mask_generators.combined_mask_generator import CombineMaskGenerator
from core.pipelines.pipeline import DatasetGenerationPipeline, BoundingBox


class HarmonizationDatasetGenerator(DatasetGenerationPipeline):
    boat_generator : ImageGenerator
    background_generator : ImageGenerator
    background_remover : BackgroundRemover
    point_extractor : PointExtractor
    harmonizer : ImageHarmonizer
    inpainter: ImageInpainter

    def generate(self) -> Tuple[Image.Image, BoundingBox]:
        boat = self.background_remover.remove(self.boat_generator.generate())
        boat = boat.resize((200, ((boat.size[1]*200)//boat.size[0])))
        bg = self.background_generator.generate()
        point = self.point_extractor.extract(bg)
        harmonized_image = self.harmonize_background_with_boat(bg, boat, point)
        inpainted_image = self.inpaint_boat_inside_background(harmonized_image, boat, point)
        return inpainted_image, BoundingBox(x=0, y=0, w=1, h=1)

    def inpaint_boat_inside_background(self, bg, boat, point):
        cropped = ImageCropper(center=point, resolution=(512, 512)).edit(bg)
        inpainting_mask = (
            CombineMaskGenerator(resolution=(512, 512)).combine(
                AlphaMaskGenerator(alpha_image=ImagePaster(center=(512 // 2, 512 // 2), patch=boat)
                                   .edit(Image.new("RGBA", (512, 512), (0, 0, 0, 0))), type=AlphaMaskGenerator.Type.border_inside, border_width=3)
            ).combine(
                AlphaMaskGenerator(alpha_image=ImagePaster(center=(512 // 2, 512 // 2), patch=boat).edit(
                    Image.new("RGBA", (512, 512), (0, 0, 0, 0))), type=AlphaMaskGenerator.Type.border_outside,
                                   border_width=3)
            ).generate()
        )
        inpainted_cropped = self.inpainter.inpaint(cropped, inpainting_mask, prompt="A boat with water trails")
        return ImagePaster(center=point, patch=inpainted_cropped).edit(bg)

    def harmonize_background_with_boat(self, bg, boat, point):
        cropped = ImageCropper(center=point, resolution=(512, 512)).edit(bg)
        bg_with_boat = ImagePaster(center=(512 // 2, 512 // 2), patch=boat).edit(cropped)
        harmonization_mask = ImagePaster(center=point, patch=AlphaMaskGenerator(
            alpha_image=ImagePaster(center=(512 // 2, 512 // 2), patch=boat).edit(
                Image.new("RGBA", (512, 512), (0, 0, 0, 0))), type=AlphaMaskGenerator.Type.inside).generate()).edit(
            Image.new("L", bg.size, 0))
        harmonization_bg = ImagePaster(center=point, patch=bg_with_boat).edit(bg)
        harmonized_image = self.harmonizer.harmonize(harmonization_bg, harmonization_mask)
        return harmonized_image


img, bbox = HarmonizationDatasetGenerator(
    boat_generator=MockImageGenerator(route="../../../data_assets/boats/without_bg/image_1-removebg-preview.png"),
    background_generator=MockImageGenerator(route="../../../data_assets/bgs/fondo3.jpg"),
    background_remover=MockBackgroundRemover(),
    point_extractor=MockPointExtractor(point=(450, 750)),
    harmonizer=MockImageHarmonizer(),
    inpainter=MockImageInpainter()
).generate()
img.save("harmonization.png")
