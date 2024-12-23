from PIL import Image

from core.dependencies.ai.generative_ai.image_generators.image_generator import ImageGenerator


class MockImageGenerator(ImageGenerator):
    def __init__(self, route : str):
        self.route = route
    def generate(self) -> Image.Image:
        return Image.open(self.route)