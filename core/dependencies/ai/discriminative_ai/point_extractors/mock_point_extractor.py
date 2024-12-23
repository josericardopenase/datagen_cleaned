from PIL import Image

from core.dependencies.ai.discriminative_ai.point_extractors.point_extractor import PointExtractor


class MockPointExtractor(PointExtractor):
    def __init__(self, point):
        self.point = point

    def extract(self, image : Image.Image):
        return self.point
