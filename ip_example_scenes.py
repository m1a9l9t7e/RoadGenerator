import numpy as np
from manim import *

from util import get_line, get_text


class HeatDiagramPlot(MovingCameraScene):
    def construct(self):
        scale = 1.2
        width = 6 * scale
        height = 6 * scale
        ax = Axes(
            x_range=[-3, 115, 20],
            y_range=[-3, 115, 20],
            x_length=width,
            y_length=height,
            x_axis_config={"numbers_to_include": np.arange(0, 101, 20)},
            y_axis_config={"numbers_to_include": np.arange(0, 101, 20)},
            tips=True,

        )
        labels = ax.get_axis_labels(
            x_label=Tex("$x_1$").scale(scale), y_label=Tex("$x_2$").scale(scale)
        )

        self.add(ax, labels)

        # constraint 1
        x_vals, y_vals = get_vals(1, 1, 100)
        print(x_vals, " ", y_vals)
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=BLUE)
        self.add(graph)

        # constraint 2
        x_vals, y_vals = get_vals(6, 9, 720)
        print(x_vals, " ", y_vals)
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=BLUE)
        self.add(graph)

        # constraint 3
        x_vals, y_vals = get_vals(0, 1, 60)
        print(x_vals, " ", y_vals)
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=BLUE)
        self.add(graph)

        # polygon
        x_vals = [0, 0, 30, 60, 100, 0]
        y_vals = [0, 60, 60, 40, 0, 0]
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals)
        self.add(graph)

        # add labels
        print(transform((21, 18), width, height))
        self.add(get_text('(2.1)', np.array([-1.8, 2.2]) * scale, scale=scale))
        self.add(get_text('(2.2)', np.array([1.3, 0.7]) * scale, scale=scale))
        self.add(get_text('(2.3)', np.array([2.8, -1.8]) * scale, scale=scale))

        self.play(
            self.camera.frame.animate.set_width(15 * scale),
            run_time=0.1
        )


def transform(coords, width, height):
    x, y = coords
    return x * width / 100, y * height / 100


def get_vals(a1, a2, b, extension=5):
    if a1 == 0:
        return [0, 100], [60, 60]
    else:
        return [int(b / a1), 0], [0, int(b / a2)]
