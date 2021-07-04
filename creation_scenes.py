from manim import *
import random
from interpolation import find_polynomials, Spline, Spline2d
from ip import GGMSTProblem
from iteration.ip_iteration import get_problem, get_intersect_matrix, convert_solution_to_join_sequence, GraphModel, convert_solution_to_graph
from util import Converter, Grid, TrackPoint, GridShowCase, draw_graph, remove_graph, make_unitary, print_2d, get_line, get_circle, get_square, add_graph
from graph import Graph, GraphSearcher
from anim_sequence import AnimationObject, AnimationSequenceScene


class MultiGraphIP(AnimationSequenceScene):
    def construct(self):
        width, height = (6, 6)
        square_size = 2
        graph_model = GraphModel(width, height, generate_intersections=True, sample_random=6 * 4)
        graph_list, helper = graph_model.get_graphs(scale=square_size, spacing=[2, 2], ratio=[6, 4])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1)
        animations = [add_graph(graph) for graph in graph_list]
        self.play_concurrent(animations)
        self.wait(3)

        track_animations_list = []  # Animations for each graph
        track_width = 0.4
        for graph in graph_list:
            gen_track_points, remove_track_points, points = generate_track_points(graph, square_size=square_size, track_width=track_width)
            interpolation_animation = interpolate_track_points_continuous(points)
            animations = gen_track_points + interpolation_animation + remove_graph(graph, animate=True) + remove_track_points
            track_animations_list.append(animations)

        self.play_concurrent(track_animations_list)
        self.wait(5)


class MultiGraph(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1.3
        track_width = 0.4
        graph_model = GraphModel(width, height, generate_intersections=False, fast=False)
        animations_list, graph_list, helper = graph_model.get_animations(scale=square_size, spacing=[2, 2])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1)
        animations = [add_graph(graph) for graph in graph_list]
        self.play_concurrent(animations)  # faster

        # self.play_concurrent(animations_list)

        # idx = 4
        # graph = graph_list[idx]
        # camera_position, camera_size = helper.get_zoomed_camera_settings(idx)
        # self.move_camera(camera_size, camera_position, duration=2)
        # gen_track_points, remove_track_points, points = generate_track_points(graph, square_size=square_size, track_width=track_width)
        # interpolation_animation = interpolate_track_points(points)
        # animations = gen_track_points + interpolation_animation + remove_track_points
        # self.play_animations(animations)
        self.wait(5)


class CircuitCreation(AnimationSequenceScene):
    def construct(self):
        width, height = (6, 6)
        square_size = 2
        track_width = 0.3

        self.play(
            self.camera.frame.animate.set_width(width * square_size * 2.1),
            run_time=0.1
        )
        self.play(
            self.camera.frame.animate.move_to((square_size * width / 2.5, square_size * height / 2.5, 0)),
            run_time=0.1
        )

        graph = Graph(width, height, scale=square_size)
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)

        graph_creation = draw_graph(graph)
        fade_out_non_unitary = make_unitary(graph)
        show_cycles = graph.init_cycles()
        # make_joins = custom_joins(graph)
        make_joins = random_joins(graph)
        gen_track_points, remove_track_points, points = generate_track_points(graph, square_size=square_size, track_width=track_width)
        interpolation_animation = interpolate_track_points_continuous(points)

        animations_list = [
            graph_creation,
            grid.get_animation_sequence(),
            fade_out_non_unitary,
            make_joins,
            gen_track_points,
            interpolation_animation,
            remove_track_points,
            remove_graph(graph)
        ]

        for animations in animations_list:
            self.play_animations(animations)

        self.wait(5)


def custom_joins(graph):
    """
    Custom joins specifically designed to resemble
    example circuit from ruleset starting from 4x4 grid
    """
    animation_sequence = []
    searcher = GraphSearcher(graph)
    joints = searcher.walk_graph()

    animation_sequence.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))

    indices = [0, 3, 2]
    operations = ['intersect', 'merge', 'intersect']

    for i, idx in enumerate(indices):
        joint = joints[idx]
        animation_sequence.append(AnimationObject(type='remove', content=joint.drawable))
        operation = operations[i]
        if operation == 'intersect':
            animation_sequence += joint.intersect()
        elif operation == 'merge':
            animation_sequence += joint.merge()
        else:
            raise ValueError('operation "{}" is undefined!'.format(operation))

    animation_sequence.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints]))
    return animation_sequence


def random_joins(graph):
    animation_sequence = []
    searcher = GraphSearcher(graph)

    while True:
        joints = searcher.walk_graph()
        if len(joints) == 0:
            break

        animation_sequence.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))
        joint = joints[0]
        animation_sequence.append(AnimationObject(type='remove', content=joint.drawable))
        if random.choice([True, False]):
            animation_sequence += joint.merge()
        else:
            animation_sequence += joint.intersect()
        animation_sequence.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints]))

    return animation_sequence


def get_random_solution(width, height):
    # Get Solution
    problem = get_problem(width, height)
    solution, status = problem.solve(_print=False)

    # Add intersections
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs=False)
    non_zero_indices = np.argwhere(intersect_matrix > 0)
    for index in range(n):
        x, y = non_zero_indices[index]
        intersect_matrix[x][y] = random.choice([0, 1])
    solution = intersect_matrix + np.array(solution)
    return solution


def get_custom_solution(width, height):
    # Get Solution
    problem = GGMSTProblem(width - 1, height - 1, [[0, 1], [1, 2], [2, 1]])
    solution, status = problem.solve(_print=False)

    # Add intersections
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs=False)
    non_zero_indices = np.argwhere(intersect_matrix > 0)
    custom_choice = [1, 1, 0] + [0] * (width * height)
    for index in range(n):
        x, y = non_zero_indices[index]
        intersect_matrix[x][y] = custom_choice[index]
    solution = intersect_matrix + np.array(solution)
    return solution


def generate_track_points(graph, square_size, track_width):
    track_points = []
    converter = Converter(graph, square_size=square_size, track_width=track_width)
    converter.extract_tour()
    nodes = converter.nodes
    nodes.append(nodes[0])

    line_drawables = []
    point_drawables = []

    for idx in range(len(nodes) - 1):
        node1 = nodes[idx]
        node2 = nodes[idx + 1]
        coord1 = node1.get_real_coords()
        coord2 = node2.get_real_coords()
        right, left, center = get_track_points(coord1, coord2, track_width)
        track_points.append((right, left, center))

        line_drawables.append(get_line(center.coords, left.coords, stroke_width=1, color=GREEN))
        line_drawables.append(get_line(center.coords, right.coords, stroke_width=1, color=GREEN))
        point_drawables.append(get_circle(right.coords, 0.04, GREEN, GREEN_E, border_width=1))
        point_drawables.append(get_circle(left.coords, 0.04, GREEN, GREEN_E, border_width=1))

    animation_sequence = [
        AnimationObject(type='play', content=[Create(line) for line in line_drawables], duration=2, bring_to_front=True),
        AnimationObject(type='play', content=[FadeIn(point) for point in point_drawables], duration=1, bring_to_front=True, wait_after=1),
        AnimationObject(type='play', content=[FadeOut(line) for line in line_drawables], duration=0.5)
    ]

    animation_sequence2 = [
        AnimationObject(type='play', content=[FadeOut(point) for point in point_drawables], duration=1, bring_to_front=True, wait_after=1),
    ]

    return animation_sequence, animation_sequence2, track_points


def alter_track_point_directions(track_points):
    for points in track_points:
        degrees_20 = 0.349066
        angle = degrees_20 * random.uniform(0, 1) - degrees_20/2
        [point.alter_direction(angle) for point in points]


def _interpolate_track_points(track_points):
    right_line_animations = []
    left_line_animations = []
    center_line_animations = []
    right1, left1, center1 = track_points[0]
    track_points = track_points[1:]
    track_points.append((right1, left1, center1))
    for right2, left2, center2 in track_points:
        px, py = find_polynomials(*(right1.as_list() + right2.as_list()))
        right_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), t_min=0, t_max=1, color=WHITE, stroke_width=2)
        right_line_animations.append(Create(right_line))
        px, py = find_polynomials(*(left1.as_list() + left2.as_list()))
        left_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), t_min=0, t_max=1, color=WHITE, stroke_width=2)
        left_line_animations.append(Create(left_line))
        px, py = find_polynomials(*(center1.as_list() + center2.as_list()))
        center_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), t_min=0, t_max=1, color=WHITE, stroke_width=2)
        dashed_line = DashedVMobject(center_line, num_dashes=5, positive_space_ratio=0.6)
        center_line_animations.append(Create(dashed_line))

        right1, left1, center1 = (right2, left2, center2)

    animation_sequence = []
    for idx in range(len(right_line_animations)):
        animation_sequence.append(AnimationObject(type='play',
                                                  content=[right_line_animations[idx], left_line_animations[idx], center_line_animations[idx]],
                                                  duration=0.5, bring_to_front=True))

    return animation_sequence


def interpolate_track_points_continuous(track_points, duration=5):
    right_spline = Spline2d()
    left_spline = Spline2d()
    center_spline = Spline2d()
    right1, left1, center1 = track_points[0]
    track_points = track_points[1:]
    track_points.append((right1, left1, center1))
    for right2, left2, center2 in track_points:
        px, py = find_polynomials(*(right1.as_list() + right2.as_list()))
        right_spline.add_polynomials(px, py)
        px, py = find_polynomials(*(left1.as_list() + left2.as_list()))
        left_spline.add_polynomials(px, py)
        px, py = find_polynomials(*(center1.as_list() + center2.as_list()))
        center_spline.add_polynomials(px, py)
        right1, left1, center1 = (right2, left2, center2)

    right_line = right_spline.get_animation()
    left_line = left_spline.get_animation()
    center_line = center_spline.get_animation(dashed=True, num_dashes=4)
    animation_sequence = [AnimationObject(type='play', content=[right_line, left_line, center_line], duration=duration, bring_to_front=True)]
    return animation_sequence


def find_center(coord1, coord2):
    x1, y1, _ = coord1
    x2, y2, _ = coord2
    return (x1 + x2) / 2, (y1 + y2) / 2, 0


def get_direction(coord1, coord2):
    """
    Calculate norm vector between two points coord1 and coord2
    """
    coord1 = np.array(coord1[:2])
    coord2 = np.array(coord2[:2])
    vec = np.subtract(coord2, coord1)
    vec_norm = vec / np.linalg.norm(vec)
    return np.array(vec_norm)


def get_orthogonal_vec(vec):
    """
    Calculate vector that is orthogonal to vec
    """
    vec_copy = np.copy(vec)
    vec_norm = vec_copy[::-1]  # change the indexing to reverse the vector to swap x and y (note that this doesn't do any copying)
    # print("change indexing: {}".format(vec_norm))
    vec_norm[0] = -vec_norm[0]
    # print("make first axis negative: {}".format(vec_norm))
    orth_vec = list(vec_norm) + [0]
    return np.array(orth_vec)


def get_track_points(coord1, coord2, track_width):
    """
    get right and left point of center of track between two coordinates
    """
    center = find_center(coord1, coord2)
    direction = get_direction(coord1, coord2)
    orth_vec = np.array(get_orthogonal_vec(direction))
    right = np.add(center, track_width * orth_vec)
    left = np.subtract(center, track_width * orth_vec)
    # return [TrackPoint(coords, direction) for coords in [right, left, center]]
    return [TrackPoint(coords, direction) for coords in [right, left, center]]


class IPCircuitCreation(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 2
        track_width = 0.4
        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))

        # solution = get_random_solution(width, height)
        solution = get_custom_solution(width, height)

        # Animate Solution
        sequence = convert_solution_to_join_sequence(solution)
        animations, graph = sequence.get_animations(square_size, (0, 0))
        gen_track_points, remove_track_points, points = generate_track_points(graph, square_size=square_size, track_width=track_width)
        interpolation_animation = interpolate_track_points_continuous(points)
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)

        animations_list = [
            grid.get_animation_sequence(),
            animations,
            gen_track_points,
            remove_graph(graph, animate=True),
            interpolation_animation,
            remove_track_points,
        ]

        for anim in animations_list:
            self.play_animations(anim)

        self.wait(4)


if __name__ == '__main__':
    # scene = CircuitCreation()
    # scene = GraphModelTest()
    # scene = IPCircuitCreation()
    scene = MultiGraphIP()
    scene.construct()
