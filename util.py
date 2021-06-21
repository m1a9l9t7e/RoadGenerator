from manim import *

from anim_sequence import AnimationObject


class Converter:
    def __init__(self, graph, square_size, track_width):
        self.graph = graph
        self.square_size = square_size
        self.track_width = track_width
        self.nodes = list()
        self.edges = list()

    def extract_tour(self):
        grid = self.graph.grid
        tour = list()
        start_node = grid[0][0]
        tour.append(start_node)
        next_node = get_next_node(start_node)

        while True:
            tour.append(next_node)
            edge = tour[-2].get_edge_to(tour[-1])
            self.edges.append(edge)
            next_node = get_next_node(tour[-1], tour[-2])
            if next_node == start_node:
                edge = tour[-1].get_edge_to(start_node)
                self.edges.append(edge)
                break

        self.nodes = tour
        print("Found tour with {} nodes and {} edges!".format(len(self.nodes), len(self.edges)))


def get_next_node(node, previous_node=None):
    if node.get_degree() != 2:
        raise ValueError("Graph not 2 factorized!")

    node1, node2 = node.adjacent_nodes

    if previous_node is None or previous_node == node2:
        return node1
    elif previous_node == node1:
        return node2


class Grid:
    def __init__(self, graph, square_size, shift, color=GREEN_C, stroke_width=0.2):
        self.graph = graph
        self.width = len(graph.grid)
        self.height = len(graph.grid[0])
        self.square_size = square_size
        self.shift = shift
        self.color = color
        self.stroke_width = stroke_width
        self.drawable = self._create_drawable()

    def _create_drawable(self):
        s = self.square_size
        shift = self.shift

        h_lines = list()
        for x in range(self.width + 1):
            line = Line(transform_coords((x, 0), shift, s), transform_coords((x, self.height), shift, s), stroke_width=self.stroke_width)
            h_lines.append(line)

        v_lines = list()
        for y in range(self.height + 1):
            line = Line(transform_coords((0, y), shift, s), transform_coords((self.width, y), shift, s), stroke_width=self.stroke_width)
            v_lines.append(line)

        lines = h_lines + v_lines
        for line in lines:
            line.set_color(self.color)

        return lines

    def get_animation_sequence(self, fade_in=True):
        animation_sequence = []
        if fade_in:
            animations = [FadeIn(drawable) for drawable in self.drawable]
            animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, bring_to_back=True))
        else:
            animation_sequence.append(AnimationObject(type='add', content=self.drawable, bring_to_back=True))

        return animation_sequence


def transform_coords(coords, shift, scale):
    x, y = coords
    x_shift, y_shift = shift
    return x_shift + x * scale, y_shift + y * scale, 0


class TrackPoint:
    def __init__(self, coords, direction):
        self.coords = coords
        self.direction = direction

    def __str__(self):
        return "coords: {}, direction: {}".format(self.coords, self.direction)

    def as_list(self):
        x, y, _ = self.coords
        dx, dy = self.direction
        return [x, y, dx, dy]
