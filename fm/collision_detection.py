import math
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

np.random.seed(192251)


def get_points(n=50):
    x = np.random.rand(n)
    y = np.random.rand(n)
    return np.reshape([x, y], [n, 2])


def draw_points(points, collision=None):
    if collision is None:
        colors = [get_cmap(20)(randrange(19))] * len(points)
    else:
        colors = ['g' if entry else 'r' for entry in collision]

    x, y = np.transpose(points, [1, 0])
    plt.scatter(x, y, s=[4] * len(points), c=colors, alpha=1)


def draw_rect(bottom_left, bottom_right, top_left, top_right):
    x_values = []
    y_values = []
    for point in [bottom_left, bottom_right, top_right, top_left, bottom_left]:
        x, y = list(point)
        x_values.append(x)
        y_values.append(y)

    plt.plot(x_values, y_values)


def bounding_box_check(points, bbox):
    bottom_left, bottom_right, top_left, top_right = bbox
    min_x, min_y = bottom_left
    max_x, max_y = top_right
    x, y = np.transpose(points, [1, 0])
    in_x = np.logical_and(min_x <= x, x <= max_x)
    in_y = np.logical_and(min_y <= y, y <= max_y)
    in_bb = np.logical_and(in_x, in_y)
    return in_bb


def rotate_points(points, radians):
    c, s = np.cos(radians), np.sin(radians)
    rotation_matrix = np.matrix([[c, s], [-s, c]])
    rotated_points = np.dot(points, rotation_matrix.T)
    return np.squeeze(np.asarray(rotated_points))


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


if __name__ == '__main__':
    # _min = -0.1
    # _max = 1.5
    # plt.xlim(_min, _max)
    # plt.ylim(_min, _max)

    # Rect, representing car outline
    bbox = [(0, 0), (0.5, 0), (0, 1), (0.5, 1)]
    bbox_rotation = math.pi * 0.1
    bbox = rotate_points(bbox, bbox_rotation)

    # Points, representing track outline
    points = get_points(n=30)

    # 1. Rotate bbox and points for easy check
    rotated_points = rotate_points(points, -bbox_rotation)
    rotated_bbox = rotate_points(bbox, -bbox_rotation)

    # 2. Check if points are in bbox
    in_bbox = bounding_box_check(rotated_points, rotated_bbox)
    # in_bbox = bounding_box_check(points, bbox)

    # Draw all
    draw_rect(*bbox)
    draw_points(points, in_bbox)

    # draw_rect(*bbox)
    # draw_points(points)
    # draw_rect(*rotated_bbox)
    # draw_points(rotated_points)

    # interval = 1
    # for i in range(int(math.pi / (interval/2))):
    #     draw_points(rotate_points(points, interval * i))
    #     draw_rect(*rotate_points(bbox, interval * i))

    # draw origin
    # plt.scatter(0, 0, s=40, color=(0, 0, 0), alpha=1)
    plt.show()


def main():
    pass
    # 1. Get Feature Model INCLUDING ZONES
    # path_to_config = os.path.join(os.getcwd(), '../ip/configs/gap.txt')
    # solution = get_solution_from_config(path_to_config, _print=False)
    # fm = FeatureModel(solution, scale=3)

    # 2. Calculate discrete points along all sold lines!
    # This depends on the track property, the selected feature, and the zone

    # 3. Get Random Rectangle and Rotate

    # 4. Check if any point is in Rectangle
