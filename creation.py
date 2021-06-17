from manim import *
import random
from util import Converter, Grid
from graph import Graph, GraphSearcher
from anim_sequence import AnimationObject, AnimationSequence


class CircuitCreation(MovingCameraScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1  # Needs to be 1 because grid and camera scale, but graph doesn't
        track_width = 0.3

        self.play(
            self.camera.frame.animate.set_width(width * square_size * 2.1),
            run_time=0.1
        )
        self.play(
            self.camera.frame.animate.move_to((square_size * width/2.5, square_size * height/2.5, 0)),
            run_time=0.1
        )

        graph_creation = self.create_graph(width, height)
        grid_fade_in = self.draw_grid(square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)
        fade_out_non_unitary = self.make_unitary()
        self.graph.init_cycles()

        make_joins = None
        if False:
            make_joins = self.custom_joins()
        else:
            make_joins = self.random_joins()

        gen_track_points, points = self.generate_track_points(square_size=square_size, track_width=track_width)

        animations_list = [graph_creation, grid_fade_in, fade_out_non_unitary, make_joins, gen_track_points]
        for animations in animations_list:
            self.play_animations(animations)

        self.wait(5)

    def create_graph(self, width=4, height=4):
        animation_sequence = []
        # Get Graph
        self.graph = Graph(width, height)
        nodes = self.graph.nodes
        edges = self.graph.edges

        node_drawables = []
        # Draw Nodes
        for node in nodes:
            circle = node.drawable
            node_drawables.append(circle)
            # self.add(circle)

        animation_sequence.append(AnimationObject(type='add', content=node_drawables))

        # Draw Edges
        line_animations = []
        for edge in edges:
            line = edge.drawable
            # self.bring_to_back(line)
            line_animations.append(Create(line))
        # self.play(*line_animations, run_time=1)
        animation_sequence.append(AnimationObject(type='play', content=line_animations, wait_after=0.5, duration=1, bring_to_back=True))

        return animation_sequence

    def make_unitary(self):
        animation_sequence = []

        drawables = self.graph.remove_all_but_unitary()
        animations = [FadeOut(drawable) for drawable in drawables]
        animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, bring_to_back=False))
        return animation_sequence

    def two_factorization(self):
        drawables_list = self.graph.two_factorization()
        for drawables in drawables_list:
            for drawable in drawables:
                self.play(FadeOut(drawable), run_time=0.1)

    def search_graph(self):
        searcher = GraphSearcher(self.graph)
        joints = searcher.walk_graph()
        for joint in joints:
            self.add(joint.drawable)

        self.wait(duration=1)

        for idx, joint in enumerate(joints):
            self.remove(joint.drawable)
            if idx >= self.graph.cycles - 1:
                break
            # animations = joint.intersect()
            animations = joint.merge()
            self.play(*animations, run_time=3)

    def custom_joins(self):
        """
        Custom joins specifically designed to resemble
        example circuit from ruleset starting from 4x4 grid
        """
        animation_sequence = []
        searcher = GraphSearcher(self.graph)
        joints = searcher.walk_graph()

        animation_sequence.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))

        indices = [0, 3, 2]
        operations = ['intersect', 'merge', 'intersect']

        for i, idx in enumerate(indices):
            joint = joints[idx]
            animation_sequence.append(AnimationObject(type='remove', content=joint.drawable))
            operation = operations[i]
            if operation == 'intersect':
                animations = joint.intersect()
            elif operation == 'merge':
                animations = joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))
            joint_animation = AnimationObject(type='play', content=animations, duration=1)
            animation_sequence.append(joint_animation)

        animation_sequence.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints]))
        return animation_sequence

    def random_joins(self):
        animation_sequence = []
        searcher = GraphSearcher(self.graph)

        while True:
            joints = searcher.walk_graph()
            if len(joints) == 0:
                break

            animation_sequence.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))
            joint = joints[0]
            animation_sequence.append(AnimationObject(type='remove', content=joint.drawable))
            if random.choice([True, False]):
                animations = joint.merge()
            else:
                animations = joint.intersect()
            joint_animation = AnimationObject(type='play', content=animations, duration=1)
            animation_sequence.append(joint_animation)
            animation_sequence.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints]))

        return animation_sequence

    def draw_grid(self, square_size=1, shift=(-0.5, -0.5), fade_in=True):
        animation_sequence = []
        grid = Grid(self.graph, square_size, shift)
        if fade_in:
            animations = [FadeIn(drawable) for drawable in grid.drawable]
            animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, bring_to_back=True))
        else:
            animation_sequence.append(AnimationObject(type='add', content=grid.drawable, bring_to_back=True))

        return animation_sequence

    def generate_track_points(self, square_size, track_width):
        animation_sequence = []
        track_points = []
        converter = Converter(self.graph, square_size=square_size, track_width=track_width)
        converter.extract_tour()
        edges = converter.edges
        nodes = converter.nodes
        nodes.append(nodes[0])

        orthogonal_lines = []
        track_point_animations = []

        for idx in range(len(nodes)-1):
            node1 = nodes[idx]
            node2 = nodes[idx+1]
            coord1 = node1.get_coords()
            coord2 = node2.get_coords()
            right, left, center = get_track_points(coord1, coord2, track_width)
            track_points.append((right, left, center))

            orthogonal_lines.append(get_line(center, left, stroke_width=1, color=GREEN))
            orthogonal_lines.append(get_line(center, right, stroke_width=1, color=GREEN))

            points_animations = [FadeIn(get_circle(right, 0.06, PURPLE, PURPLE_E)),
                                 FadeIn(get_circle(left, 0.06, MAROON, MAROON_E))]
            track_point_animations.append(points_animations)

        animation_sequence.append(AnimationObject(type='play', content=[Create(line) for line in orthogonal_lines], duration=2, bring_to_front=True))
        for track_point_anim in track_point_animations:
            animation_sequence.append(AnimationObject(type='play', content=track_point_anim, duration=0.3, bring_to_front=True))

        animation_sequence.append(AnimationObject(type='play', content=[FadeOut(line) for line in orthogonal_lines], duration=0.5))

        return animation_sequence, track_points

    def play_animations(self, sequence):
        for animation in sequence:
            if animation.bring_to_back or animation.bring_to_front:
                content = animation.content
                if animation.type == 'play':
                    content = [c.mobject for c in animation.content]
                if animation.bring_to_front:
                    self.bring_to_front(*content)
                if animation.bring_to_back:
                    self.bring_to_back(*content)
            if animation.type == 'add':
                self.add(*animation.content)
            if animation.type == 'remove':
                self.remove(*animation.content)
            elif animation.type == 'play':
                self.play(*animation.content, run_time=animation.duration)

            self.wait(animation.wait_after)


def find_center(coord1, coord2):
    x1, y1, _ = coord1
    x2, y2, _ = coord2
    return (x1 + x2) / 2, (y1 + y2) / 2, 0


def get_orthogonal_vec(coord1, coord2):
    """
    Calculates vector that is orthogonal to
    vector between two points coord1 and coord2
    """
    coord1 = np.array(coord1[:2])
    coord2 = np.array(coord2[:2])
    vec = np.subtract(coord2, coord1)
    # print("vector: {}".format(vec))
    vec_norm = vec / np.linalg.norm(vec)
    # print("normalized vector: {}".format(vec_norm))
    vec_norm = vec_norm[::-1]  # change the indexing to reverse the vector to swap x and y (note that this doesn't do any copying)
    # print("change indexing: {}".format(vec_norm))
    vec_norm[0] = -vec_norm[0]
    # print("make first axis negative: {}".format(vec_norm))
    orth_vec = list(vec_norm) + [0]
    return orth_vec


def get_track_points(coord1, coord2, track_width):
    """
    get right and left point of center of track between two coordinates
    """
    center_point = find_center(coord1, coord2)
    orth_vec = np.array(get_orthogonal_vec(coord1, coord2))
    right_point = np.add(center_point, track_width * orth_vec)
    left_point = np.subtract(center_point, track_width * orth_vec)
    return right_point, left_point, center_point


def get_circle(coords, radius, color, secondary_color):
    circle = Dot(point=coords, radius=radius)
    circle.set_fill(color, opacity=1)
    circle.set_stroke(secondary_color, width=1)
    return circle


def get_line(coord1, coord2, stroke_width=1.0, color=WHITE):
    line = Line(coord1, coord2, stroke_width=stroke_width)
    line.set_color(color)
    return line


if __name__ == '__main__':
    scene = CircuitCreation()
    scene.construct()


# Plan:
# 1. Draw Grid
# 2. Draw Graph and show Joins
# 3. Show how track points are found
# 4. Draw in interpolated points that from track
