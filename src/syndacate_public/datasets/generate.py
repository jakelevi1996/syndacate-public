import os
import numpy as np
from jutility import plotting, util
from syndacate_public.datasets import shapes, grapheme, words, image

class ImageSampler:
    def sample(self, rng: np.random.Generator) -> list[np.ndarray]:
        raise NotImplementedError()

    def get_summary(self) -> str:
        raise NotImplementedError()

    def save_dataset(self):
        raise NotImplementedError()

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir

    def get_full_path(self, loading=True) -> str:
        return util.get_full_path(
            "%s_%s.npz" % (type(self).__name__.lower(), self.get_summary()),
            self.output_dir,
            loading=loading,
        )

class Syndacate(ImageSampler):
    def __init__(
        self,
        seed:               int,
        num_elements:       int,
        max_num_parts:      None | int,
        flip_parts:         bool,
        max_num_objects:    int,
        object_replacement: bool,
        align_sets:         bool,
        width:              int,
        height:             int,
        upsample_ratio:     int,
        output_dir:         str,
    ):
        self.part_pose_dim = shapes.Line().pose.size
        self.grapheme_names = self.get_grapheme_names()
        self.object_pose_dim = (
            len(self.grapheme_names)
            + len(grapheme.Grapheme.get_pose_names())
        )
        if max_num_parts is None:
            max_num_parts_per_object = max(
                len(grapheme.grapheme_dict[s]().parts)
                for s in self.grapheme_names
            )
            max_num_parts = max_num_parts_per_object * max_num_objects

        self.seed               = seed
        self.num_elements       = num_elements
        self.max_num_parts      = max_num_parts
        self.flip_parts         = flip_parts
        self.max_num_objects    = max_num_objects
        self.object_replacement = object_replacement
        self.align_sets         = align_sets
        self.width              = width
        self.height             = height
        self.upsample_ratio     = upsample_ratio
        self.set_output_dir(output_dir)

    @classmethod
    def get_grapheme_names(cls) -> list[str]:
        return "A E F I L M N W X Z".split()

    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.get_grapheme_names())

    def sample(self, rng: np.random.Generator) -> list[np.ndarray]:
        num_objects = rng.integers(1, self.max_num_objects + 1)
        grapheme_inds = rng.choice(
            len(self.grapheme_names),
            size=num_objects,
            replace=self.object_replacement,
        )
        graphemes: list[grapheme.Grapheme] = []
        object_poses = []
        for i in grapheme_inds.tolist():
            grapheme_type = grapheme.grapheme_dict[self.grapheme_names[i]]
            g = grapheme_type(
                wideness    = rng.uniform(0.5,  1   ),
                italic      = rng.uniform(0,    0.5 ),
                rotation    = rng.uniform(-0.5, 0.5 ),
                scale       = rng.uniform(0.2,  0.5 ),
                x           = rng.uniform(-1,   1   ),
                y           = rng.uniform(-1,   1   ),
                thickness   = rng.uniform(5,    10  ),
                brightness  = rng.uniform(0.5,  1   ),
            )
            graphemes.append(g)
            class_one_hot = np.zeros(len(self.grapheme_names))
            class_one_hot[i] = 1
            full_pose = np.concatenate([class_one_hot, g.pose], axis=0)
            object_poses.append(full_pose)

        parts = [part for g in graphemes for part in g.parts]
        part_object_inds = [
            i for i, g in enumerate(graphemes) for part in g.parts
        ]
        if len(parts) > self.max_num_parts:
            part_inds = rng.choice(
                len(parts),
                size=self.max_num_parts,
                replace=False,
            )
            parts = [parts[j] for j in part_inds]
            part_object_inds = [part_object_inds[j] for j in part_inds]
        if self.flip_parts:
            flipped_parts = []
            for p in parts:
                flipped_parts.append(p.flip())

            parts = parts + flipped_parts
            part_object_inds = part_object_inds + part_object_inds
        if self.align_sets:
            object_poses = [object_poses[i] for i in part_object_inds]

        part_pose_array = np.stack(
            [part.pose for part in parts],
            dtype=np.float32,
        )
        object_pose_array = np.stack(object_poses, axis=0, dtype=np.float32)
        im = image.Image(
            *sorted(parts),
            width=self.width,
            height=self.height,
            upsample_ratio=self.upsample_ratio,
        )
        image_array = im.get_array()
        return [image_array, part_pose_array, object_pose_array]

    def get_summary(self) -> str:
        return util.abbreviate_dictionary(
            {
                "seed":                 self.seed,
                "num_elements":         self.num_elements,
                "max_num_parts":        self.max_num_parts,
                "flip_parts":           self.flip_parts,
                "max_num_objects":      self.max_num_objects,
                "object_replacement":   self.object_replacement,
                "align_sets":           self.align_sets,
                "width":                self.width,
                "height":               self.height,
                "upsample_ratio":       self.upsample_ratio,
            },
            key_abbreviations={
                "seed":                 "s",
                "num_elements":         "n",
                "max_num_parts":        "p",
                "flip_parts":           "f",
                "max_num_objects":      "o",
                "object_replacement":   "r",
                "align_sets":           "a",
                "width":                "w",
                "height":               "h",
                "upsample_ratio":       "u",
            },
        )

    def save_dataset(self):
        rng = np.random.default_rng(self.seed)

        num_channels = 1
        images_shape = [
            self.num_elements,
            num_channels,
            self.height,
            self.width,
        ]
        part_poses_shape = [
            self.num_elements,
            self.max_num_parts,
            self.part_pose_dim,
        ]
        object_poses_shape = [
            self.num_elements,
            self.max_num_objects,
            self.object_pose_dim,
        ]

        if self.flip_parts:
            part_poses_shape[1] *= 2
        if part_poses_shape[1] > 1:
            part_poses_shape[1] += 1
        if object_poses_shape[1] > 1:
            object_poses_shape[1] += 1
        if self.align_sets:
            object_poses_shape[1] = part_poses_shape[1]

        images       = np.zeros(images_shape,       dtype=np.uint8)
        part_poses   = np.zeros(part_poses_shape,   dtype=np.float32)
        object_poses = np.zeros(object_poses_shape, dtype=np.float32)

        for i in util.progress(range(self.num_elements), "Creating image "):
            image_array, part_pose_array, object_pose_array = self.sample(rng)

            part_inds = rng.choice(
                part_poses.shape[1],
                size=part_pose_array.shape[0],
                replace=False,
            )
            if self.align_sets:
                object_inds = part_inds
            else:
                object_inds = rng.choice(
                    object_poses.shape[1],
                    size=object_pose_array.shape[0],
                    replace=False,
                )

            images[i, :, :, :]              = image_array
            part_poses[  i, part_inds,   :] = part_pose_array
            object_poses[i, object_inds, :] = object_pose_array

        np.savez_compressed(
            self.get_full_path(loading=False),
            images=images,
            part_poses=part_poses,
            object_poses=object_poses,
            n=self.num_elements,
        )

class Words(ImageSampler):
    def __init__(
        self,
        seed:               int,
        num_elements:       int,
        max_num_objects:    int,
        width:              int,
        height:             int,
        upsample_ratio:     int,
        output_dir:         str,
    ):
        self.word_names = self.get_word_names()
        self.object_pose_dim = (
            len(self.word_names)
            + len(words.Word.get_pose_names())
        )

        self.seed               = seed
        self.num_elements       = num_elements
        self.max_num_objects    = max_num_objects
        self.width              = width
        self.height             = height
        self.upsample_ratio     = upsample_ratio
        self.set_output_dir(output_dir)

    @classmethod
    def get_word_names(cls) -> list[str]:
        return "Ai Hi Me Aim Fan Him Fail Film Lime Mile".split()

    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.get_word_names())

    def sample(self, rng: np.random.Generator) -> list[np.ndarray]:
        num_objects = rng.integers(1, self.max_num_objects + 1)
        word_inds = rng.choice(
            len(self.word_names),
            size=num_objects,
            replace=True,
        )
        word_list: list[words.Word] = []
        object_poses = []
        for i in word_inds.tolist():
            word_type = words.word_dict[self.word_names[i]]
            g = word_type(
                wideness    = rng.uniform(0.5,  1   ),
                italic      = rng.uniform(0,    0.5 ),
                rotation    = rng.uniform(-0.5, 0.5 ),
                scale       = rng.uniform(0.2,  0.5 ),
                x           = rng.uniform(-1,   1   ),
                y           = rng.uniform(-1,   1   ),
                thickness   = rng.uniform(3,    5   ),
                brightness  = rng.uniform(0.5,  1   ),
            )
            word_list.append(g)
            class_one_hot = np.zeros(len(self.word_names))
            class_one_hot[i] = 1
            full_pose = np.concatenate([class_one_hot, g.pose], axis=0)
            object_poses.append(full_pose)

        object_pose_array = np.stack(object_poses, axis=0, dtype=np.float32)
        im = image.Image(
            *sorted(word_list),
            width=self.width,
            height=self.height,
            upsample_ratio=self.upsample_ratio,
        )
        image_array = im.get_array()
        return [image_array, object_pose_array]

    def get_summary(self) -> str:
        return util.abbreviate_dictionary(
            {
                "seed":                 self.seed,
                "num_elements":         self.num_elements,
                "max_num_objects":      self.max_num_objects,
                "width":                self.width,
                "height":               self.height,
                "upsample_ratio":       self.upsample_ratio,
            },
            key_abbreviations={
                "seed":                 "s",
                "num_elements":         "n",
                "max_num_objects":      "o",
                "width":                "w",
                "height":               "h",
                "upsample_ratio":       "u",
            },
        )

    def save_dataset(self):
        rng = np.random.default_rng(self.seed)

        num_channels = 1
        images_shape = [
            self.num_elements,
            num_channels,
            self.height,
            self.width,
        ]
        object_poses_shape = [
            self.num_elements,
            self.max_num_objects,
            self.object_pose_dim,
        ]

        if object_poses_shape[1] > 1:
            object_poses_shape[1] += 1

        images       = np.zeros(images_shape,       dtype=np.uint8)
        object_poses = np.zeros(object_poses_shape, dtype=np.float32)

        for i in util.progress(range(self.num_elements), "Creating image "):
            image_array, object_pose_array = self.sample(rng)

            object_inds = rng.choice(
                object_poses.shape[1],
                size=object_pose_array.shape[0],
                replace=False,
            )

            images[i, :, :, :]              = image_array
            object_poses[i, object_inds, :] = object_pose_array

        np.savez_compressed(
            self.get_full_path(loading=False),
            images=images,
            object_poses=object_poses,
            n=self.num_elements,
        )

class Squares(ImageSampler):
    def __init__(
        self,
        seed:               int,
        num_elements:       int,
        max_num_objects:    int,
        width:              int,
        height:             int,
        upsample_ratio:     int,
        output_dir:         str,
    ):
        self.object_pose_dim = sum(
            shapes.Square.get_pose_names().count(k)
            for k in ["x", "y"]
        )

        self.seed               = seed
        self.num_elements       = num_elements
        self.max_num_objects    = max_num_objects
        self.width              = width
        self.height             = height
        self.upsample_ratio     = upsample_ratio
        self.set_output_dir(output_dir)

    def sample(self, rng: np.random.Generator) -> list[np.ndarray]:
        num_objects = rng.integers(1, self.max_num_objects + 1)
        squares = [
            shapes.Square(
                x=rng.uniform(-1, 1),
                y=rng.uniform(-1, 1),
                length=5/100,
                thickness=0,
            )
            for i in range(num_objects)
        ]
        object_poses = [
            [p for k in ["x", "y"] for p in s.get_pose_elements(k)]
            for s in squares
        ]
        object_pose_array = np.array(object_poses, dtype=np.float32)
        im = image.Image(
            *sorted(squares),
            width=self.width,
            height=self.height,
            upsample_ratio=self.upsample_ratio,
        )
        image_array = im.get_array()
        return [image_array, object_pose_array]

    def get_summary(self) -> str:
        return util.abbreviate_dictionary(
            {
                "seed":                 self.seed,
                "num_elements":         self.num_elements,
                "max_num_objects":      self.max_num_objects,
                "width":                self.width,
                "height":               self.height,
                "upsample_ratio":       self.upsample_ratio,
            },
            key_abbreviations={
                "seed":                 "s",
                "num_elements":         "n",
                "max_num_objects":      "o",
                "width":                "w",
                "height":               "h",
                "upsample_ratio":       "u",
            },
        )

    def save_dataset(self):
        rng = np.random.default_rng(self.seed)

        num_channels = 1
        images_shape = [
            self.num_elements,
            num_channels,
            self.height,
            self.width,
        ]
        object_poses_shape = [
            self.num_elements,
            self.max_num_objects,
            self.object_pose_dim,
        ]

        if object_poses_shape[1] > 1:
            object_poses_shape[1] += 1

        images       = np.zeros(images_shape,       dtype=np.uint8)
        object_poses = np.zeros(object_poses_shape, dtype=np.float32)

        for i in util.progress(range(self.num_elements), "Creating image "):
            image_array, object_pose_array = self.sample(rng)

            object_inds = rng.choice(
                object_poses.shape[1],
                size=object_pose_array.shape[0],
                replace=False,
            )

            images[i, :, :, :]              = image_array
            object_poses[i, object_inds, :] = object_pose_array

        np.savez_compressed(
            self.get_full_path(loading=False),
            images=images,
            object_poses=object_poses,
            n=self.num_elements,
        )
