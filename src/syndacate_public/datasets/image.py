import numpy as np
import PIL.Image
import PIL.ImageDraw
from jutility import plotting, util
from syndacate_public.datasets import objects

class Image:
    def __init__(
        self,
        *visual_objects: objects.VisualObject,
        width=100,
        height=100,
        upsample_ratio=10,
    ):
        upsample_width  = int(width  * upsample_ratio)
        upsample_height = int(height * upsample_ratio)

        image = PIL.Image.new("L", (upsample_width, upsample_height))
        draw  = PIL.ImageDraw.Draw(image)

        for i in visual_objects:
            i.render(draw, upsample_ratio, *image.size)

        canvas_size = (width, height)
        self.pil_image = image.resize(canvas_size)

    def get_array(self):
        return np.array(self.pil_image)

    def subplot(self, title=None):
        return plotting.Subplot(
            plotting.ImShow(self.get_array(), vmin=0, vmax=255),
            title=title,
        )

    def save(self, output_name, dir_name=None):
        full_path = util.get_full_path(output_name, dir_name, "png")
        self.pil_image.save(full_path)
