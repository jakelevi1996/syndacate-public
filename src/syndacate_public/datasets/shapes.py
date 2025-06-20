import numpy as np
import PIL.ImageDraw
from jutility import plotting
from syndacate_public.datasets import objects

class Line(objects.Terminal):
    @classmethod
    def get_pose_names(cls) -> list[str]:
        return ["x", "y", "x", "y", "thickness", "brightness"]

    def get_default_pose(self):
        return {"thickness": 1, "brightness": 1}

    def get_depth(self) -> float:
        return self.pose[self.get_pose_names().index("brightness")]

    def flip(self) -> "objects.VisualObject":
        x1, y1, x2, y2, t, b = self.pose
        return Line(x=[x2, x1], y=[y2, y1], thickness=t, brightness=b)

    def plot(self, show_marker=True, **kwargs):
        [thickness]  = self.get_pose_elements("thickness")
        [brightness] = self.get_pose_elements("brightness")
        if show_marker:
            kwargs.update(marker="o", ms=3*thickness)

        line = plotting.Line(
            self.get_pose_elements("x"),
            self.get_pose_elements("y"),
            lw=thickness,
            a=np.clip(brightness, 0, 1),
            **kwargs,
        )
        return [line]

    def render(
        self,
        draw: PIL.ImageDraw.ImageDraw,
        upsample_ratio: int,
        width:  int,
        height: int,
    ):
        [thickness]  = self.get_pose_elements("thickness")
        [brightness] = self.get_pose_elements("brightness")
        draw.line(
            xy=self.get_image_coords(width, height).flatten().tolist(),
            fill=int(brightness * 255),
            width=int(thickness * upsample_ratio),
        )

class FilledShape(objects.Terminal):
    @classmethod
    def _get_num_points(cls) -> int:
        raise NotImplementedError()

    @classmethod
    def get_pose_names(cls) -> list[str]:
        n = cls._get_num_points()
        return (["x", "y"] * n) + ["thickness", "brightness", "fill"]

    def get_default_pose(self):
        return {"thickness": 1, "brightness": 1, "fill": 1}

    def plot(self, **kwargs):
        xy = (
            self.get_pose_elements("x"),
            self.get_pose_elements("y"),
        )
        [thickness]  = self.get_pose_elements("thickness")
        [brightness] = self.get_pose_elements("brightness")
        [fill]       = self.get_pose_elements("fill")
        p1 = plotting.Polygon(
            *xy,
            lw=5 * thickness,
            fill=False,
            a=brightness,
            **kwargs,
        )
        p2 = plotting.Polygon(
            *xy,
            lw=0,
            fill=True,
            a=fill,
            **kwargs,
        )
        return [p1, p2]

    def render(
        self,
        draw: PIL.ImageDraw.ImageDraw,
        upsample_ratio: int,
        width:  int,
        height: int,
    ):
        [thickness]  = self.get_pose_elements("thickness")
        [brightness] = self.get_pose_elements("brightness")
        [fill]       = self.get_pose_elements("fill")
        draw.polygon(
            xy=self.get_image_coords(width, height).flatten().tolist(),
            fill=int(fill * 255),
            outline=int(brightness * 255),
            width=int(thickness * upsample_ratio),
        )

class Triangle(FilledShape):
    @classmethod
    def _get_num_points(cls) -> int:
        return 3

class Quadrilateral(FilledShape):
    @classmethod
    def _get_num_points(cls) -> int:
        return 4

class Square(objects.VisualObject):
    @classmethod
    def get_pose_names(cls) -> list[str]:
        return ["x", "y", "length", "thickness", "brightness", "fill"]

    def get_default_pose(self):
        return {"length": 1, "thickness": 1, "brightness": 1, "fill": 1}

    def get_depth(self) -> float:
        return self.pose[self.get_pose_names().index("brightness")]

    def _generate_parts(self) -> None:
        x, y, length, t, b, f = self.pose
        hl = length / 2
        part = Quadrilateral(
            x=[x-hl, x+hl, x+hl, x-hl],
            y=[y-hl, y-hl, y+hl, y+hl],
            thickness=t,
            brightness=b,
            fill=f,
        )
        self._add_part(part)

class Turbine(objects.VisualObject):
    @classmethod
    def get_pose_names(cls) -> list[str]:
        return ["x", "y", "scale", "thickness", "brightness"]

    def get_default_pose(self):
        return {"scale": 1, "thickness": 1, "brightness": 1}

    def _generate_parts(self) -> None:
        x, y, s, t, b = self.pose
        for i in range(3):
            part = Line(x=[0, 0], y=[0, 1], thickness=t, brightness=b)
            part.rotate(i * 2 * np.pi / 3)
            part.translate(x, y)
            part.scale(s, s)
            self._add_part(part)
