import numpy as np
import PIL.Image
import PIL.ImageDraw
from jutility import plotting, util

class VisualObject:
    def __init__(self, **pose_kwargs):
        self._pose_ind_dict = self._get_pose_ind_dict()
        self.pose = np.zeros(self.get_pose_dim())
        pose_dict = self.get_default_pose()
        pose_dict.update(pose_kwargs)
        for k, v in pose_dict.items():
            self.set_pose_elements(k, v)

        self.parts: list[VisualObject] = []
        self._generate_parts()

    @classmethod
    def from_pose(cls, pose: np.ndarray) -> "VisualObject":
        kwargs = {
            k: pose[v]
            for k, v in cls._get_pose_ind_dict().items()
        }
        return cls(**kwargs)

    @classmethod
    def get_pose_names(cls) -> list[str]:
        raise NotImplementedError()

    @classmethod
    def get_pose_dim(cls) -> int:
        return len(cls.get_pose_names())

    @classmethod
    def _get_pose_ind_dict(cls) -> dict[str, list[int]]:
        pose_names = cls.get_pose_names()
        pose_ind_dict = {s: [] for s in set(pose_names)}
        for i, s in enumerate(pose_names):
            pose_ind_dict[s].append(i)

        return pose_ind_dict

    def _generate_parts(self) -> None:
        raise NotImplementedError()

    def get_depth(self) -> float:
        raise NotImplementedError()

    def get_default_pose(self):
        return dict()

    def get_pose_elements(self, name) -> list[float]:
        return self.pose[self._pose_ind_dict[name]].tolist()

    def set_pose_elements(self, name, pose):
        self.pose[self._pose_ind_dict[name]] = pose

    def _add_part(self, part: "VisualObject"):
        self.parts.append(part)

    def get_range(self):
        range_list = [part.get_range() for part in self.parts]
        xmin, xmax, ymin, ymax = zip(*range_list)
        return min(xmin), max(xmax), min(ymin), max(ymax)

    def get_radius(self):
        return max(part.get_radius() for part in self.parts)

    def normalise(self):
        xmin, xmax, ymin, ymax = self.get_range()
        self.translate(-(xmin + xmax) / 2, -(ymin + ymax) / 2)

        r_inv = 1 / self.get_radius()
        self.scale(r_inv, r_inv)

    def plot(self, **kwargs) -> list[plotting.Plottable]:
        return [
            plottable
            for part in self.parts
            for plottable in part.plot(**kwargs)
        ]

    def render(
        self,
        draw: PIL.ImageDraw.ImageDraw,
        upsample_ratio: int,
        width:  int,
        height: int,
    ):
        for part in self.parts:
            part.render(draw, upsample_ratio, width, height)

    def copy(self) -> "VisualObject":
        return self.from_pose(self.pose)

    def translate(self, x: float, y: float):
        for part in self.parts:
            part.translate(x, y)

    def scale(self, x: float, y: float):
        for part in self.parts:
            part.scale(x, y)

    def rotate(self, theta_rad: float):
        for part in self.parts:
            part.rotate(theta_rad)

    def shear(self, x: float, y: float):
        for part in self.parts:
            part.shear(x, y)

    def flip(self) -> "VisualObject":
        raise NotImplementedError()

    def __repr__(self):
        pose_kwargs = {
            s: self.get_pose_elements(s)
            for s in self._pose_ind_dict.keys()
        }
        return util.format_type(type(self), **pose_kwargs)

    def __lt__(self, other: "VisualObject"):
        return self.get_depth() < other.get_depth()

class Terminal(VisualObject):
    def _generate_parts(self):
        return

    def plot(self, **kwargs):
        raise NotImplementedError()

    def render(
        self,
        draw: PIL.ImageDraw.ImageDraw,
        upsample_ratio: int,
        width:  int,
        height: int,
    ):
        raise NotImplementedError()

    def get_coords(self):
        x = self.get_pose_elements("x")
        y = self.get_pose_elements("y")
        return np.stack([x, y], axis=1)

    def set_coords(self, coords):
        x, y = np.split(coords, 2, axis=1)
        self.set_pose_elements("x", x.flatten())
        self.set_pose_elements("y", y.flatten())

    def get_image_coords(self, width: int, height: int):
        canvas_size = [width, height]
        relative_canvas_size = np.array(canvas_size) / max(canvas_size)

        image_coords = self.get_coords() * np.array([1, -1])
        image_coords += relative_canvas_size
        image_coords /= 2
        image_coords *= max(canvas_size)

        return image_coords

    def get_range(self):
        coords = self.get_coords()
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        return xmin, xmax, ymin, ymax

    def get_radius(self):
        c2 = np.square(self.get_coords())
        norms = np.sqrt(np.sum(c2, axis=1))
        return np.max(norms)

    def translate(self, x: float, y: float):
        self.set_coords(self.get_coords() + np.array([x, y]))

    def scale(self, x: float, y: float):
        self.set_coords(self.get_coords() * np.array([x, y]))

    def rotate(self, theta_rad: float):
        cos = np.cos(theta_rad)
        sin = np.sin(theta_rad)
        rotation_matrix = [
            [ cos, sin],
            [-sin, cos],
        ]
        self.set_coords(self.get_coords() @ np.array(rotation_matrix).T)

    def shear(self, x: float, y: float):
        shear_matrix = [
            [1, x        ],
            [y, 1 + (x*y)],
        ]
        self.set_coords(self.get_coords() @ np.array(shear_matrix).T)
