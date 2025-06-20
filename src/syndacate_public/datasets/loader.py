from numpy.lib.npyio import NpzFile
import torch
from syndacate_public.datasets import grapheme, generate

class NpzLoader:
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        raise NotImplementedError()

    def load_images(self, npz_file: NpzFile) -> torch.Tensor:
        return torch.tensor(npz_file["images"], dtype=torch.float32) / 255.0

    def load_parts(self, npz_file: NpzFile) -> torch.Tensor:
        return torch.tensor(npz_file["part_poses"], dtype=torch.float32)

    def load_objects(self, npz_file: NpzFile) -> torch.Tensor:
        return torch.tensor(npz_file["object_poses"], dtype=torch.float32)

class ImageLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        return self.load_images(npz_file)

class PartLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        return self.load_parts(npz_file)

class LineSingleEndpointLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        full_pose = self.load_parts(npz_file)
        return full_pose[..., :2]

class LineBothEndpointsLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        full_pose = self.load_parts(npz_file)
        return full_pose[..., :4]

class ObjectLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        return self.load_objects(npz_file)

class SingleObjectLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        return self.load_objects(npz_file).squeeze(dim=-2)

class GraphemeSingleClassLoader(NpzLoader):
    def load(self, npz_file: NpzFile) -> torch.Tensor:
        class_one_hot, _ = torch.split(
            self.load_objects(npz_file).squeeze(dim=-2),
            [
                generate.Syndacate.get_num_classes(),
                grapheme.Grapheme.get_pose_dim(),
            ],
            dim=-1,
        )
        class_int_labels = class_one_hot.argmax(dim=-1, keepdim=False)
        return class_int_labels
