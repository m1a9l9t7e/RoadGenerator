import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class Constraint:
    def __init__(self, x, y, derivative):
        self.x = x
        self.y = y
        self.derivative = derivative


def find_polynomials(x1, y1, dx1, dy1, x2, y2, dx2, dy2):
    constraints = [
        Constraint(0, x1, 0),
        Constraint(0, dx1, 1),
        Constraint(1, x2, 0),
        Constraint(1, dx2, 1),
        Constraint(0, 0, 2),
        Constraint(1, 0, 2),
    ]

    a, b = get_equation_system(constraints)
    x = np.linalg.solve(a, b)
    polynomial_x = np.poly1d(x)

    constraints = [
        Constraint(0, y1, 0),
        Constraint(0, dy1, 1),
        Constraint(1, y2, 0),
        Constraint(1, dy2, 1),
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


def spline_unit_circle(radius=2):
    theta = 2 * np.pi * np.linspace(0, 1, 5)[:2]
    # theta = np.linspace(0, 1, 5)

    points = np.c_[np.cos(theta) * radius, np.sin(theta) * radius]
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print("theta:\n{} ".format(theta))
    print("points:\n{} ".format(points))

    # cs = CubicSpline(theta, points, bc_type='periodic')
    cs = CubicSpline(theta, points)

    # print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
    # print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 0)[0], cs(0, 0)[1]))
    # ds / dx = 0.0
    # ds / dy = 1.0

    xs = 2 * np.pi * np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(points[:, 0], points[:, 1], 'o', label='data')
    ax.plot(np.cos(xs), np.sin(xs), label='true')

    # Interpolation here
    ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')

    ax.axes.set_aspect('equal')
    ax.legend(loc='center')
    plt.show()


if __name__ == '__main__':
    # px, py = find_polynomials(0, 0, 1, 0, 1, 1, 0, 1)
    # px, py = find_polynomials(0, 0, 1, 0, 0.75, 0.75, 0, 1)
    # plot(px, py, 20)
    spline_unit_circle()
