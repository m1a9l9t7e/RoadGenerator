import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from fm.model import FeatureModel
from ip.iteration import get_solution_from_config

np.random.seed(192251)


def get_points(n=50):
    # 1. Get Feature Model INCLUDING ZONES
    path_to_config = os.path.join(os.getcwd(), '../ip/configs/demo.txt')
    solution = get_solution_from_config(path_to_config, _print=False)
    fm = FeatureModel(solution, scale=3)

    # 2. Calculate discrete points along all sold lines!
    lines = fm.get_collision_lines(track_width=0.42)
    collision_points = []
    for line in lines:
        collision_points += line.get_points()
    draw_points(collision_points)


def get_random_points(n=50):
    x = np.random.rand(n)
    y = np.random.rand(n)
    return np.reshape([x, y], [n, 2])


def draw_points(points, collision=None):
    if collision is None:
        colors = [1] * len(points)
    else:
        colors = ['r' if entry else 'g' for entry in collision]

    x, y = np.transpose(points, [1, 0])
    plt.scatter(x, y, s=[2] * len(points), c=colors, alpha=1)


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


def random_rect(min_coords, max_coords, scale=1):
    bbox = [(0, 0), (0.5*scale, 0), (0, 1*scale), (0.5*scale, 1*scale)]
    bbox_rotation = math.pi * random.uniform(0, 2)
    bbox = rotate_points(bbox, bbox_rotation)

    x_offset = random.uniform(min_coords[0], max_coords[0])
    y_offset = random.uniform(min_coords[1], max_coords[1])
    bbox = [(x + x_offset, y + y_offset) for (x, y) in bbox]
    return bbox, bbox_rotation


def detect():
    # Rect, representing car outline
    bbox = [(0, 0), (0.5, 0), (0, 1), (0.5, 1)]
    bbox_rotation = math.pi * 0.1
    bbox = rotate_points(bbox, bbox_rotation)

    # Points, representing track outline
    points = get_random_points(n=50)

    # 1. Rotate bbox and points for easy check
    rotated_points = rotate_points(points, -bbox_rotation)
    rotated_bbox = rotate_points(bbox, -bbox_rotation)

    # 2. Check if points are in bbox
    in_bbox = bounding_box_check(rotated_points, rotated_bbox)

    # Draw all
    draw_rect(*bbox)
    draw_points(points, in_bbox)

    plt.show()


def points_in_rect(rect, rect_rotation, points):
    # 1. Rotate rect and points for easy check
    rotated_points = rotate_points(points, -rect_rotation)
    rotated_bbox = rotate_points(rect, -rect_rotation)

    # 2. Check if points are in bbox
    in_rect = bounding_box_check(rotated_points, rotated_bbox)
    return in_rect


def main(num_cars=100):
    scale = 3

    # 1. Get Feature Model INCLUDING ZONES
    path_to_config = os.path.join(os.getcwd(), '../ip/configs/demo.txt')
    solution = get_solution_from_config(path_to_config, _print=False)
    fm = FeatureModel(solution, scale=scale)

    # 2. Calculate discrete points along all sold lines!
    lines = fm.get_collision_lines(track_width=0.42)
    collision_points = []
    for line in lines:
        collision_points += line.get_points()

    # 3. Get Random Rectangle and Rotate
    car_list = []
    for i in range(num_cars):
        rect, rotation = random_rect((0.5, 0.5), ((fm.graph.width - 1) * scale, (fm.graph.height - 1) * scale))
        car_list.append((rect, rotation))

    # 4. Check if any point is in Rectangle
    complete_collision = None
    for (rect, rotation) in car_list:
        if complete_collision is None:
            complete_collision = points_in_rect(rect, rotation, collision_points)
        else:
            complete_collision = np.logical_or(points_in_rect(rect, rotation, collision_points), complete_collision)

    # 5. Draw all
    for (rect, rotation) in car_list:
        draw_rect(*rect)

    draw_points(collision_points, complete_collision)

    plt.show()


if __name__ == '__main__':
    main()





