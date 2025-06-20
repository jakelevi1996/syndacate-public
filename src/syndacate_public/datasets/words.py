import numpy as np
from syndacate_public.datasets import grapheme

class Word(grapheme.Grapheme):
    def get_default_pose(self):
        return {"wideness": 1, "scale": 1, "thickness": 3, "brightness": 1}

    def _generate_parts(self) -> None:
        w, i, r, s, x, y, t, b = self.pose
        line_thickness = t * s
        max_displacement = 1 - s

        for grapheme_name in type(self).__name__.upper():
            grapheme_type = grapheme.grapheme_dict[grapheme_name]
            g = grapheme_type(thickness=line_thickness, brightness=b)

            xmin, xmax, ymin, ymax = g.get_range()
            g_scale = 1 / (np.sqrt(2) * max(abs(ymin), abs(ymax)))
            g.scale(g_scale, g_scale)

            self._add_part(g)
            self.translate(-2, 0)

        self.normalise()
        self.scale(w, 1)
        self.shear(i, 0)
        self.normalise()
        self.rotate(r)
        self.scale(s, s)
        self.translate(x * max_displacement, y * max_displacement)

word_dict: dict[str, type[Word]] = {
    s: type(s, tuple([Word]), dict())
    for s in [
        "Ai",
        "Hi",
        "Me",
        "Aim",
        "Fan",
        "Him",
        "Fail",
        "Film",
        "Lime",
        "Mile",
        "Hello",
        "Hollie",
    ]
}
word_names = sorted(word_dict.keys())
