import numpy as np
import matplotlib.pyplot as plt
from manim import *


class Constraint:
    def __init__(self, x, y, derivative):
        self.x = x
        self.y = y
        self.derivative = derivative


def find_polynomials(x1, y1, dx1, dy1, x2, y2, dx2, dy2):
    distance = np.linalg.norm([x1 - x2, y1 - y2])
    constraints = [
        Constraint(0, x1, 0),
        Constraint(0, dx1 * distance, 1),
        Constraint(1, x2, 0),
        Constraint(1, dx2 * distance, 1),
        Constraint(0, 0, 2),
        Constraint(1, 0, 2),
    ]

    a, b = get_equation_system(constraints)
    x = np.linalg.solve(a, b)
    polynomial_x = np.poly1d(x)

    constraints = [
        Constraint(0, y1, 0),
        Constraint(0, dy1 * distance, 1),
        Constraint(1, y2, 0),
        Constraint(1, dy2 * distance, 1),
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


def plot(polynomial_x, polynomial_y, n):
    x = [polynomial_x(z) for z in np.arange(0, 1, 1/n)]
    y = [polynomial_y(z) for z in np.arange(0, 1, 1 / n)]
    plt.scatter(x, y)
    plt.show()


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

    def get_animation(self, dashed=False, num_dashes=10):
        # parametric_function = ParametricFunction(function=lambda t: (self.spline_x(t), self.spline_y(t), 0), t_min=0, t_max=len(self), color=WHITE, stroke_width=2)
        parametric_function = ParametricFunction(function=lambda t: (self.spline_x(t), self.spline_y(t), 0), t_min=0, t_max=2, color=WHITE, stroke_width=2)
        if dashed:
            parametric_function = DashedVMobject(parametric_function, num_dashes=num_dashes, positive_space_ratio=0.6)
        animation = Create(parametric_function)
        return animation


if __name__ == '__main__':
    px, py = find_polynomials(0, 0, 1, 0, 1, 1, 0, 1)
    plot(px, py, 20)
