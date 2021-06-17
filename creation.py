from manim import *
from util import Converter, Grid
from graph import Graph, GraphSearcher


class CircuitCreation(MovingCameraScene):
    def construct(self):
        self.create_graph(10, 10)
        self.make_unitary()
        self.graph.init_cycles()
        # self.custom_joins()
        self.random_joins()
        self.wait(2)

    def create_graph(self, width=4, height=4):
        # Get Graph
        self.graph = Graph(width, height)
        nodes = self.graph.nodes
        edges = self.graph.edges
        self.play(
            # Set the size with the width of a object
            self.camera.frame.animate.set_width(width * 1.7),
            # Move the camera to the object
            self.camera.frame.animate.move_to((width/2.5, height/2.5, 0))
        )

        # Draw Nodes
        for node in nodes:
            circle = node.drawable
            self.add(circle)

        # Draw Edges
        animations = []
        for edge in edges:
            line = edge.drawable
            self.bring_to_back(line)
            animations.append(Create(line))

        self.play(*animations, run_time=1)

    def make_unitary(self):
        drawables = self.graph.remove_all_but_unitary()
        animations = [FadeOut(drawable) for drawable in drawables]
        self.play(*animations, run_time=0.5)

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
        searcher = GraphSearcher(self.graph)
        joints = searcher.walk_graph()
        for joint in joints:
            self.add(joint.drawable)

        self.wait(duration=1)

        indices = [0, 3, 2]
        operations = ['intersect', 'merge', 'intersect']

        for i, idx in enumerate(indices):
            joint = joints[idx]
            self.remove(joint.drawable)
            operation = operations[i]
            if operation == 'intersect':
                animations = joint.intersect()
            elif operation == 'merge':
                animations = joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))
            self.play(*animations, run_time=3)

        for joint in joints:
            self.remove(joint.drawable)

    def random_joins(self):
        searcher = GraphSearcher(self.graph)

        while True:
            joints = searcher.walk_graph()
            if len(joints) == 0:
                break

            for joint in joints:
                self.add(joint.drawable)

            self.wait(duration=1)

            # join first joint
            joint = joints[0]
            self.remove(joint.drawable)
            animations = joint.merge()
            self.play(*animations, run_time=3)

            for joint in joints:
                self.remove(joint.drawable)


class DrawGrid(MovingCameraScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 2
        track_width = 0.2
        self.play(
            # Set the size with the width of a object
            self.camera.frame.animate.set_width(width * 1.7),
            # Move the camera to the object
            self.camera.frame.animate.move_to((width/2.5, height/2.5, 0)),
            run_time=0.1
        )

        self.graph = self.init_graph(width, height)
        self.merge_cycles()
        # self.draw_grid(square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size) # TODO FIX
        self.draw_grid(square_size=1, shift=np.array([-0.5, -0.5]) * 1)  # TODO FIX
        self.generate_track_points(square_size=square_size, track_width=0.2)
        self.wait(duration=5)

    def init_graph(self, width, height):
        g = Graph(width, height)
        g.remove_all_but_unitary()
        g.init_cycles()
        return g

    def merge_cycles(self):
        # find joints and merge until single cycle
        searcher = GraphSearcher(self.graph)
        while True:
            joints = searcher.walk_graph()
            if len(joints) == 0:
                break
            # join first joint
            joint = joints[0]
            # joint.merge()
            joint.intersect()

    def draw_grid(self, square_size=1, shift=(-0.5, -0.5)):
        grid = Grid(self.graph, square_size, shift)
        self.add(*grid.drawable)

    def generate_track_points(self, square_size, track_width):
        track_points = []
        converter = Converter(self.graph, square_size=square_size, track_width=track_width)
        converter.extract_tour()
        edges = converter.edges
        nodes = converter.nodes
        nodes.append(nodes[0])

        for idx in range(len(nodes)-1):
            node1 = nodes[idx]
            node2 = nodes[idx+1]
            coord1 = node1.get_coords()
            coord2 = node2.get_coords()
            right, left = get_track_points(coord1, coord2, track_width)
            track_points.append((right, left))

            self.add(edges[idx % len(edges)].drawable)
            self.add(node1.drawable)
            self.add(node2.drawable)
            self.add(get_simple_circle(right, 0.06, PURPLE, PURPLE_E))
            self.add(get_simple_circle(left, 0.06, MAROON, MAROON_E))
            self.wait(0.1)

        return track_points


def get_simple_circle(coords, radius, color, secondary_color):
    circle = Dot(point=coords, radius=radius)
    circle.set_fill(color, opacity=1)
    circle.set_stroke(secondary_color, width=4)
    return circle


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
    center = find_center(coord1, coord2)
    orth_vec = np.array(get_orthogonal_vec(coord1, coord2))
    right_point = np.add(center, track_width * orth_vec)
    left_point = np.subtract(center, track_width * orth_vec)
    return right_point, left_point


if __name__ == '__main__':
    scene = DrawGrid()
    scene.construct()
