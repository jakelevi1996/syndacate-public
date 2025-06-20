from syndacate_public.datasets import objects, shapes

class Grapheme(objects.VisualObject):
    @classmethod
    def get_pose_names(cls) -> list[str]:
        return [
            "wideness",
            "italic",
            "rotation",
            "scale",
            "x",
            "y",
            "thickness",
            "brightness",
        ]

    def get_default_pose(self):
        return {"wideness": 1, "scale": 1, "thickness": 10, "brightness": 1}

    def get_depth(self) -> float:
        return self.pose[self.get_pose_names().index("brightness")]

    def _generate_parts(self) -> None:
        w, i, r, s, x, y, t, b = self.pose
        line_thickness = t * s
        max_displacement = 1 - s

        for [x1, y1], [x2, y2] in self.get_canonical_line_coords():
            line = shapes.Line(
                x=[x1, x2],
                y=[y1, y2],
                thickness=line_thickness,
                brightness=b,
            )
            self._add_part(line)

        self.normalise()
        self.scale(w, 1)
        self.shear(i, 0)
        self.normalise()
        self.rotate(r)
        self.scale(s, s)
        self.translate(x * max_displacement, y * max_displacement)

    def get_canonical_line_coords(self):
        raise NotImplementedError()

class A(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0,    0  ], [0.5,  1  ]],
            [[1,    0  ], [0.5,  1  ]],
            [[0.25, 0.5], [0.75, 0.5]],
        ]

class E(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[1, 1  ], [0, 1  ]],
            [[0, 0  ], [0, 1  ]],
            [[0, 0  ], [1, 0  ]],
            [[0, 0.5], [1, 0.5]],
        ]

class F(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[1, 1  ], [0, 1  ]],
            [[0, 0  ], [0, 1  ]],
            [[0, 0.5], [1, 0.5]],
        ]

class H(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0, 0  ], [0, 1  ]],
            [[1, 0  ], [1, 1  ]],
            [[0, 0.5], [1, 0.5]],
        ]

class I(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0.5, 0], [0.5, 1]],
        ]

class L(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0, 0], [0, 1]],
            [[0, 0], [1, 0]],
        ]

class M(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0,   0   ], [0.25, 1]],
            [[0.5, 0.25], [0.25, 1]],
            [[0.5, 0.25], [0.75, 1]],
            [[1  , 0   ], [0.75, 1]],
        ]

class N(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0, 0], [0, 1]],
            [[1, 0], [0, 1]],
            [[1, 0], [1, 1]],
        ]

class O(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0, 0], [0, 1]],
            [[1, 1], [0, 1]],
            [[1, 1], [1, 0]],
            [[0, 0], [1, 0]],
        ]

class W(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0,   1   ], [0.25, 0]],
            [[0.5, 0.75], [0.25, 0]],
            [[0.5, 0.75], [0.75, 0]],
            [[1,   1   ], [0.75, 0]],
        ]

class X(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0, 0], [1, 1]],
            [[0, 1], [1, 0]],
        ]

class Z(Grapheme):
    def get_canonical_line_coords(self):
        return [
            [[0, 1], [1, 1]],
            [[0, 0], [1, 1]],
            [[0, 0], [1, 0]],
        ]

def get_all_types() -> list[type[Grapheme]]:
    return [A, E, F, H, I, L, M, N, O, W, X, Z]

grapheme_dict = {
    g.__name__: g
    for g in get_all_types()
}
grapheme_names = sorted(grapheme_dict.keys())
