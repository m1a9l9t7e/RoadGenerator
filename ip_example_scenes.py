import numpy as np
from manim import *
from termcolor import colored

from interpolation import interpolate_single, Constraint, get_equation_system
from util import get_line, get_text, get_circle, TrackPoint

LIGHT_MODE = False

plot_size = 3


class HeatDiagramPlot(MovingCameraScene):
    def construct(self):
        if LIGHT_MODE:
            self.camera.background_color = WHITE

        scale = 2
        # width = 6 * scale
        # height = 6 * scale

        width = plot_size * 2 * scale
        height = plot_size * 2 * scale
        # ax = Axes(
        #     x_range=[-3, 115, 20],
        #     y_range=[-3, 115, 20],
        #     x_length=width,
        #     y_length=height,
        #     x_axis_config={"numbers_to_include": np.arange(0, 101, 20)},
        #     y_axis_config={"numbers_to_include": np.arange(0, 101, 20)},
        #     tips=True,
        #
        # )
        ax = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=width,
            y_length=height,
            axis_config={
                'color': BLACK if LIGHT_MODE else WHITE,
                'stroke_width': 5,
            },
            x_axis_config={"numbers_to_include": np.arange(0, 6, 1), "font_size": 60, 'color': BLACK if LIGHT_MODE else WHITE},
            y_axis_config={"numbers_to_include": np.arange(0, 6, 1), "font_size": 60, 'color': BLACK if LIGHT_MODE else WHITE},
            tips=True,
        )
        self.add(ax)

        self.add(get_text('$x_1$', np.array([6 - plot_size, -0.4 - plot_size]) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))
        self.add(get_text('$x_2$', np.array([-0.4 - plot_size, 6 - plot_size]) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))

        # constraint 1
        # -x + 5
        x_vals, y_vals = get_vals(1, 1, 100)
        print("({}, {}) to ({}, {})".format(x_vals[0], y_vals[0], x_vals[1], y_vals[1]))
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=BLUE, add_vertex_dots=False)
        self.add(graph)

        # constraint 2
        # -0.666667 x + 3.9
        # x_vals, y_vals = get_vals(2, 3, 235)
        x_vals, y_vals = get_vals(2, 3, 235)
        print("({}, {}) to ({}, {})".format(x_vals[0], y_vals[0], x_vals[1], y_vals[1]))
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=BLUE, add_vertex_dots=False)
        self.add(graph)

        # constraint 3
        # 2.5
        x_vals, y_vals = get_vals(0, 1, 60)
        print("({}, {}) to ({}, {})".format(x_vals[0], y_vals[0], x_vals[1], y_vals[1]))
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, line_color=BLUE, add_vertex_dots=False)
        self.add(graph)

        # polygon
        # x_vals = [0, 0, 30, 60, 100, 0]
        # y_vals = [0, 60, 60, 40, 0, 0]
        # graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals)
        # self.add(graph)
        #
        # # add labels
        # print(transform((21, 18), width, height))
        # self.add(get_text('(2.4)', np.array([-1.8, 2.2]) * scale, scale=scale))
        # self.add(get_text('(2.6)', np.array([1.3, 0.7]) * scale, scale=scale))
        # self.add(get_text('(2.5)', np.array([2.8, -1.8]) * scale, scale=scale))

        # polygon
        x_vals = [0, 0, 2.1, 3.3, 5, 0]
        y_vals = [0, 2.5, 2.5, 1.7, 0, 0]
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals, add_vertex_dots=False, line_color=GREEN if LIGHT_MODE else YELLOW)
        self.add(graph)

        # convex hull
        x_vals = [0, 0, 2, 5, 0]
        y_vals = [0, 2, 2, 0, 0]

        lines = custom_dashed(x_vals, y_vals, scale)
        for line in lines:
            self.add(line)

        # add labels
        # print(transform((21, 18), width, height))
        # self.add(get_text('(2.4)', (np.array([1, 4.8]) - _scale) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))
        # self.add(get_text('(2.5)', (np.array([4, 2.9]) - _scale) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))
        # self.add(get_text('(2.6)', (np.array([5.5, 0.8]) - _scale) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))

        self.add(get_text('$x_1 + x_2 \\leq 5$', (np.array([1.5, 4.8]) - plot_size) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))
        self.add(get_text('$x_2 \\leq 2.5$', (np.array([4, 2.8]) - plot_size) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))
        self.add(get_text('$2x_1 + 3x_2 \\leq 11.75$', (np.array([6.5, 0.8]) - plot_size) * scale, scale=1.5, color=BLACK if LIGHT_MODE else WHITE))

        self.play(
            self.camera.frame.animate.set_width(15 * scale),
            run_time=0.1
        )

        for x in range(0, 5, 1):
            for y in range(0, 5, 1):
                if y == 0 and x <= 5:
                    pass
                elif y == 1 and x <= 3:
                    pass
                elif y == 2 and x <= 2:
                    pass
                else:
                    continue
                _x, _y = transform((x, y), scale, width, height)
                self.add(get_circle((_x, _y, 0), 0.08, RED_C, RED_E))


def transform(coords, scale, width, height):
    x, y = coords
    return (x - plot_size) * scale, (y - plot_size) * scale


def get_vals(a1, a2, b, divisor=20):
    if a1 == 0:
        # return [0, a2 / divisor], [a2 / divisor, a2 / divisor]
        return [0, 6], [2.5, 2.5]
    else:
        return [int(b / a1) / divisor, 0], [0, int(b / a2) / divisor]


# def get_vals(a1, a2, b, extension=5):
#     if a1 == 0:
#         return [0, 100], [60, 60]
#     else:
#         return [int(b / a1), 0], [0, int(b / a2)]

def custom_dashed(x_vals, y_vals, scale):
    lines = []
    shift_x, shift_y = (-plot_size * scale, -plot_size * scale)
    for index in range(len(x_vals)-1):
        x1 = x_vals[index]
        y1 = y_vals[index]
        x2 = x_vals[index+1]
        y2 = y_vals[index+1]
        distance = np.linalg.norm([x1 - x2, y1 - y2])
        px, py = find_polynomials_simple(x1, y1, x2, y2)
        line = ParametricFunction(function=lambda t: (px(t) * scale + shift_x, py(t) * scale + shift_y, 0), color=RED, stroke_width=6)
        # if index == 4:
        #     line = DashedVMobject(line)
        # else:
        line = DashedVMobject(line, num_dashes=int(distance * 5))
        lines.append(line)

    return lines


def find_polynomials_simple(x1, y1, x2, y2):
    constraints = [
        Constraint(0, x1, 0),
        Constraint(1, x2, 0),
    ]

    a, b = get_equation_system(constraints)
    x = np.linalg.solve(a, b)
    polynomial_x = np.poly1d(x)

    constraints = [
        Constraint(0, y1, 0),
        Constraint(1, y2, 0),
    ]

    a, b = get_equation_system(constraints)
    x = np.linalg.solve(a, b)
    polynomial_y = np.poly1d(x)

    return polynomial_x, polynomial_y


if __name__ == '__main__':
    scene = HeatDiagramPlot()
    scene.construct()
