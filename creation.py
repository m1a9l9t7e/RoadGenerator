from manim import *
import random
from interpolation import find_polynomials
from util import Converter, Grid, TrackPoint, GridShowCase
from graph import Graph, GraphSearcher, GraphModel
from anim_sequence import AnimationObject, AnimationSequenceScene


class MultiGraph(AnimationSequenceScene):
    def construct(self):
        num_graphs = 12
        width, height = (4, 4)
        square_size = 1.3
        track_width = 0.4
        graph_width = square_size * width
        graph_height = square_size * height

        helper = GridShowCase(num_graphs, (graph_width, graph_height))
        camera_position, camera_size = helper.get_global_camera_settings()
        self.play(
            self.camera.frame.animate.set_width(camera_size[0] * 1.3),
            run_time=0.1
        )
        self.play(
            self.camera.frame.animate.move_to((camera_position[0], camera_position[1], 0)),
            run_time=0.1
        )

        graphs = []
        animations_list = []

        for index in range(num_graphs):
            shift = helper.get_element_coords(index)
            graph = Graph(width, height, scale=square_size, shift=shift)
            graphs.append(graph)
            animations_list.append(draw_graph(graph))

        self.play_concurrent(animations_list)
        self.wait(5)


class CircuitCreation(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1.3  # Needs to be 1 because grid and camera scale, but graph doesn't
        track_width = 0.4

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
        interpolation_animation = interpolate_track_points(points)

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


def draw_graph(graph):
    animation_sequence = []
    node_drawables = [FadeIn(node.drawable) for node in graph.nodes]
    edge_drawables = [Create(edge.drawable) for edge in graph.edges]
    animation_sequence.append(AnimationObject(type='play', content=node_drawables, duration=1, bring_to_front=True))
    animation_sequence.append(AnimationObject(type='play', content=edge_drawables, duration=1, bring_to_back=True))
    return animation_sequence


def remove_graph(graph):
    drawables = [node.drawable for node in graph.nodes] + [edge.drawable for edge in graph.edges]
    animations = [FadeOut(drawable) for drawable in drawables]
    # return [AnimationObject(type='play', content=animations, duration=1)]
    return [AnimationObject(type='remove', content=drawables)]


def make_unitary(graph):
    animation_sequence = []

    drawables = graph.remove_all_but_unitary()
    animations = [FadeOut(drawable) for drawable in drawables]
    animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, bring_to_back=False))
    return animation_sequence


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


def interpolate_track_points(track_points):
    left_line_animations = []
    right_line_animations = []
    right1, left1, center1 = track_points[0]
    track_points = track_points[1:]
    track_points.append((right1, left1, center1))
    for right2, left2, center2 in track_points:
        # print("Connect: {} and {}".format(right1, right2))
        px, py = find_polynomials(*(right1.as_list() + right2.as_list()))
        right_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), t_min=0, t_max=1, color=WHITE, stroke_width=1)
        right_line_animations.append(Create(right_line))
        px, py = find_polynomials(*(left1.as_list() + left2.as_list()))
        left_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), t_min=0, t_max=1, color=WHITE, stroke_width=1)
        left_line_animations.append(Create(left_line))
        # dashed_line = DashedVMobject(line) for center line?

        right1, left1, center1 = (right2, left2, center2)

    animation_sequence = []
    for idx in range(len(right_line_animations)):
        animation_sequence.append(AnimationObject(type='play',
                                                  content=[right_line_animations[idx], left_line_animations[idx]],
                                                  duration=0.5, bring_to_front=True))

    return animation_sequence


# def two_factorization(graph):
#     drawables_list = graph.two_factorization()
#     for drawables in drawables_list:
#         for drawable in drawables:
#             self.play(FadeOut(drawable), run_time=0.1)
#
#
# def search_graph(graph):
#     searcher = GraphSearcher(graph)
#     joints = searcher.walk_graph()
#     for joint in joints:
#         self.add(joint.drawable)
#
#     self.wait(duration=1)
#
#     for idx, joint in enumerate(joints):
#         self.remove(joint.drawable)
#         if idx >= graph.cycles - 1:
#             break
#         # animations = joint.intersect()
#         animations = joint.merge()
#         self.play(*animations, run_time=3)


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


def get_circle(coords, radius, color, secondary_color, border_width=2):
    circle = Dot(point=coords, radius=radius)
    circle.set_fill(color, opacity=1)
    circle.set_stroke(secondary_color, width=border_width)
    return circle


def get_line(coord1, coord2, stroke_width=1.0, color=WHITE):
    line = Line(coord1, coord2, stroke_width=stroke_width)
    line.set_color(color)
    return line

class GraphModelTest(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1  # Needs to be 1 because grid and camera scale, but graph doesn't

        self.play(
            self.camera.frame.animate.set_width(width * square_size * 2.1),
            run_time=0.1
        )
        self.play(
            self.camera.frame.animate.move_to((square_size * width / 2.5, square_size * height / 2.5, 0)),
            run_time=0.1
        )

        base_graph = Graph(width, height)
        base_graph.remove_all_but_unitary()
        base_graph.init_cycles()

        model = GraphModel(base_graph)
        graph = model.iterate_all_possible_tours()[1]
        # graph_list = model.iterate_all_possible_tours()[1:]

        animation_sequence = []

        node_animations = [FadeIn(node.drawable) for node in graph.nodes]
        edge_drawables = [edge.drawable for edge in graph.edges]
        edge_animations = [Create(edge.drawable) for edge in graph.edges]
        animation_sequence.append(AnimationObject(type='play', content=node_animations, duration=1, bring_to_front=True))
        # animation_sequence.append(AnimationObject(type='play', content=edge_animations, duration=1, bring_to_back=True))
        # animation_sequence.append(AnimationObject(type='remove', content=edge_drawables))
        # animation_sequence.append(AnimationObject(type='add', content=edge_drawables))

        self.play_animations(animation_sequence)
        self.add(*edge_drawables)
        self.bring_to_back(*edge_drawables)

        self.wait(2)

        # drawables = []
        # for edge in graph.edges:
        #     # print(edge)
        #     drawables.append(Create(edge.drawable))
        #
        # self.play_animations([AnimationObject(type='play', content=drawables, bring_to_back=True, duration=1)])
        self.wait(5)


class LineTest(MovingCameraScene):
    def construct(self):
        px, py = find_polynomials(0, 0, 1, 0, 0.1, 0.1, 0, 1)
        _line_x = ParametricFunction(function=lambda t: (t - 2, px(t), 0), t_min=0, t_max=2, color=WHITE)
        _line_y = ParametricFunction(function=lambda t: (t, py(t), 0), t_min=0, t_max=2, color=WHITE)
        _line = ParametricFunction(function=lambda t: (px(t) + 2, py(t), 0), t_min=0, t_max=0.1, color=WHITE)
        label_x = Text("fx(z)")
        label_x.next_to(_line_x, DOWN)
        label_y = Text("fy(z)")
        label_y.next_to(_line_y, DOWN)
        label = Text("fx,y(z)")
        label.next_to(_line, DOWN)
        # self.play(Create(label_x), Create(label_y), Create(label))
        self.add(label_x, label_y, label)
        self.play(Create(_line_x))
        self.play(Create(_line_y))
        self.play(Create(_line), run_time=2)
        self.wait(5)


class CircleTest(MovingCameraScene):
    def construct(self):
        # scale_list = [0.1, 0.5, 0.75, 1, 2, 10, 100]
        scale_list = [0.1, 0.5, 0.75, 1, 2]
        x_pos = 0
        for scale_idx, scale in enumerate(scale_list):
            circle_points = [np.array([0, 0]) * scale, np.array([1, 1]) * scale, np.array([0, 2]) * scale, np.array([-1, 1]) * scale]
            circle_directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            width = scale * 5 + 1 if scale_idx > 0 else 2.2
            spacing = 0 if scale_idx > 0 else -width * 0.9 + 1
            current_pos = x_pos + width/2
            self.play(
                self.camera.frame.animate.set_width(width),
                run_time=0.5
            )
            self.play(
                self.camera.frame.animate.move_to((current_pos, scale/1.5, 0)),
                run_time=1
            )
            label = Text("r={}".format(scale), size=0.5)
            label.next_to((current_pos, 0, 0), DOWN)
            self.add(label)

            x_pos += spacing + width

            for idx in range(len(circle_points)):
                next_idx = (idx + 1) % len(circle_points)
                point1, direction1 = (circle_points[idx], circle_directions[idx])
                point2, direction2 = (circle_points[next_idx], circle_directions[next_idx])
                px, py = find_polynomials(*point1, *direction1, *point2, *direction2)
                _line = ParametricFunction(function=lambda t: (current_pos + px(t), py(t), 0), t_min=0, t_max=1, color=WHITE, stroke_width=2 if scale < 10 else (np.log10(scale) + 2) * 5)
                self.play(Create(_line), run_time=0.5)

        self.wait(3)


if __name__ == '__main__':
    # scene = CircuitCreation()
    # scene = GraphModelTest()
    scene = MultiGraph()
    scene.construct()
