from abc import abstractmethod, ABC
from PIL import Image

class ImageEditor(ABC):
    @abstractmethod
    def edit(self, img : Image.Image):
        ...