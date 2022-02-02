import math

from manim import *
from anim_sequence import AnimationObject
import numpy as np


class Constraint:
    def __init__(self, x, y, derivative):
        self.x = x
        self.y = y
        self.derivative = derivative


def find_polynomials(x1, y1, dx1, dy1, x2, y2, dx2, dy2):
    distance = np.linalg.norm([x1 - x2, y1 - y2])

    # angle = get_angle((dx1, dy1), (dx2, dy2))
    # if angle < 0.01:
    #     angle = 1
    # else:
    #     angle /= (math.pi / 2)
    #     if angle == 0.5:
    #         angle = 0.95
    #
    # print(angle)
    # if np.isnan(angle):
    #     angle = 1
    angle = 1

    constraints = [
        Constraint(0, x1, 0),
        Constraint(0, dx1 * distance * angle, 1),
        Constraint(1, x2, 0),
        Constraint(1, dx2 * distance * angle, 1),
        Constraint(0, 0, 2),
        Constraint(1, 0, 2),
    ]

    a, b = get_equation_system(constraints)
    x = np.linalg.solve(a, b)
    polynomial_x = np.poly1d(x)

    constraints = [
        Constraint(0, y1, 0),
        Constraint(0, dy1 * distance * angle, 1),
        Constraint(1, y2, 0),
        Constraint(1, dy2 * distance * angle, 1),
        Constraint(0, 0, 2),
        Constraint(1, 0, 2),
    ]

    a, b = get_equation_system(constraints)
    x = np.linalg.solve(a, b)
    polynomial_y = np.poly1d(x)

    return polynomial_x, polynomial_y


def get_equation_system(constraints):
    a = []
    b = []
    for c in constraints:
        a.append(calculate_polynomial_values(c.x, len(constraints), c.derivative))
        b.append(c.y)

    return a, b


def calculate_polynomial_values(x, rank, derivative, descending=True):
    values = []
    for i in range(rank):
        value = None
        factor = 1

        if descending:
            exponent = rank-i-1
        else:
            exponent = i

        for _ in range(derivative):
            factor *= exponent
            if exponent == 0:
                break
            exponent -= 1

        # print("c={} * x={}^e={}".format(factor, x, exponent))
        value = factor * x ** exponent

        values.append(value)

    return values


class InterpolatedLine:
    def __init__(self, px, py):
        self.py = py
        self.px = px
        self.length = self.get_length()

    def get_points(self, n=None):
        if n is None:
            n = int(self.length * 7)
        points = []
        for value in np.linspace(0, 1, n):
            points.append(np.array([self.px(value), self.py(value)]))
        return points

    def get_length(self):
        length = 0
        points = self.get_points(100)
        for index in range(len(points) - 1):
            length += np.linalg.norm(points[index] - points[index+1])
        return length


class Spline:
    def __init__(self, polynomials=None):
        if polynomials is None:
            self.polynomials = list()
        else:
            self.polynomials = polynomials

    def __len__(self):
        return len(self.polynomials)

    def add_polynomial(self, p):
        self.polynomials.append(p)

    def __call__(self, z):
        z *= len(self)
        index = min(int(z), len(self)-1)
        # print("p({}) -> p{}({})".format(z, index, z-index))
        return self.polynomials[index](z-index)


class Spline2d:
    def __init__(self, spline_x=None, spline_y=None):
        self.spline_x = Spline() if spline_x is None else spline_x
        self.spline_y = Spline() if spline_y is None else spline_y

    def __len__(self):
        return len(self.spline_x)

    def add_polynomials(self, px, py):
        self.spline_x.add_polynomial(px)
        self.spline_y.add_polynomial(py)

    def get_splines(self):
        return self.spline_x, self.spline_y

    def get_animation(self, dashed=False, num_dashes=5, _color=WHITE, stroke_width=2):
        # parametric_function = ParametricFunction(function=lambda t: (self.spline_x(t), self.spline_y(t), 0), t_min=0, t_max=len(self), color=WHITE, stroke_width=2)
        parametric_function = ParametricFunction(function=lambda t: (self.spline_x(t), self.spline_y(t), 0), color=_color, stroke_width=stroke_width)
        if dashed:
            parametric_function = DashedVMobject(parametric_function, num_dashes=num_dashes * len(self), dashed_ratio=0.6)
        animation = Create(parametric_function)
        return animation


def interpolate_single(start, end):
    px, py = find_polynomials(*(start.as_list() + end.as_list()))
    return px, py


def interpolate_track_points_piece_wise(track_points):
    right_line_polynomials = []
    left_line_polynomials = []
    center_line_polynomials = []
    right1, left1, center1 = track_points[0]
    track_points = track_points[1:]
    track_points.append((right1, left1, center1))
    for right2, left2, center2 in track_points:
        px, py = find_polynomials(*(right1.as_list() + right2.as_list()))
        right_line_polynomials.append((px, py))
        px, py = find_polynomials(*(left1.as_list() + left2.as_list()))
        left_line_polynomials.append((px, py))
        px, py = find_polynomials(*(center1.as_list() + center2.as_list()))
        center_line_polynomials.append((px, py))
        right1, left1, center1 = (right2, left2, center2)

    return right_line_polynomials, left_line_polynomials, center_line_polynomials


def get_interpolation_animation_piece_wise(track_points, colors=None, z_index=None, stroke_width=2):
    r_polynomials, l_polynomials, c_polynomials = interpolate_track_points_piece_wise(track_points)
    if colors is None:
        colors = [WHITE] * len(r_polynomials)
    animation_sequence = []
    for idx in range(len(r_polynomials)):
        px, py = r_polynomials[idx]
        right_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=colors[idx], stroke_width=stroke_width)
        px, py = l_polynomials[idx]
        left_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=colors[idx], stroke_width=stroke_width)
        px, py = c_polynomials[idx]
        center_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=colors[idx], stroke_width=stroke_width)
        center_line = DashedVMobject(center_line, num_dashes=5, dashed_ratio=0.6)
        animation_sequence.append(AnimationObject(type='play',
                                                  content=[Create(right_line), Create(left_line), Create(center_line)],
                                                  duration=0.25, bring_to_front=True, z_index=z_index))
    return animation_sequence


def get_interpolation_animation_continuous(points, duration=5, _color=WHITE, stroke_width=2):
    right_spline = Spline2d()
    left_spline = Spline2d()
    center_spline = Spline2d()
    right, left, center = interpolate_track_points_piece_wise(points)
    for index in range(len(right)):
        px, py = right[index]
        right_spline.add_polynomials(px, py)
        px, py = left[index]
        left_spline.add_polynomials(px, py)
        px, py = center[index]
        center_spline.add_polynomials(px, py)

    right_line = right_spline.get_animation(_color=_color, stroke_width=stroke_width)
    left_line = left_spline.get_animation(_color=_color, stroke_width=stroke_width)
    center_line = center_spline.get_animation(dashed=True, num_dashes=5, _color=_color, stroke_width=stroke_width)
    animation_sequence = [AnimationObject(type='play', content=[right_line, left_line, center_line], duration=duration, bring_to_front=True)]
    return animation_sequence


def get_angle(vector1, vector2):
    # unit_vector_1 = vector1 / np.linalg.norm(vector1)
    # unit_vector_2 = vector2 / np.linalg.norm(vector2)
    # dot_product = np.dot(unit_vector_1, unit_vector_2)
    # angle = np.arccos(dot_product)

    x1, y1 = vector1
    x2, y2 = vector2
    angle = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    return angle


def interpolate_line(track_points):
    line_polynomials = []
    track_point1 = track_points[0]
    track_points = track_points[1:]
    for track_point2 in track_points:
        px, py = find_polynomials(*(track_point1.as_list() + track_point2.as_list()))
        line_polynomials.append((px, py))
        track_point1 = track_point2

    return line_polynomials


def get_interpolation_animation_line(track_points, colors=None, z_index=None, stroke_width=2):
    polynomials = interpolate_line(track_points)
    if colors is None:
        colors = [WHITE] * len(polynomials)
    animation_sequence = []
    for idx in range(len(polynomials)):
        px, py = polynomials[idx]
        line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=colors[idx], stroke_width=stroke_width)
        animation_sequence.append(AnimationObject(type='add', content=line, z_index=z_index))
    return animation_sequence


def get_interpolation_animation_line_continuous(points, duration=5, _color=WHITE, stroke_width=2):
    spline = Spline2d()
    polynomials = interpolate_line(points)
    for (px, py) in range(len(polynomials)):
        spline.add_polynomials(px, py)

    line = spline.get_animation(_color=_color, stroke_width=stroke_width)
    animation_sequence = [AnimationObject(type='play', content=[line], duration=duration, bring_to_front=True)]
    return animation_sequence


if __name__ == '__main__':
    px, py = find_polynomials(0, 0, 1, 0, 1, 1, 0, 1)
    pass
