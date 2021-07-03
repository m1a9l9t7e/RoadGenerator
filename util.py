from manim import *
import math
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
        # print("Found tour with {} nodes and {} edges!".format(len(self.nodes), len(self.edges)))


def get_next_node(node, previous_node=None):
    if node.get_degree() != 2:
        raise ValueError("Graph not 2 factorized!")

    node1, node2 = node.adjacent_nodes

    if previous_node is None or previous_node == node2:
        return node1
    elif previous_node == node1:
        return node2


class Grid:
    def __init__(self, graph, square_size, shift, color=GREEN_C, stroke_width=0.4):
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
        self.direction = direction / np.linalg.norm(direction)

    def __str__(self):
        return "coords: {}, direction: {}".format(self.coords, self.direction)

    def alter_direction(self, angle):
        """
        alter direction by adding angle in radian
        :param angle:
        :return:
        """
        current_angle = np.arctan2(*self.direction)
        new_angle = current_angle + angle
        direction = [np.sin(new_angle), np.cos(new_angle)]
        self.direction = np.round(direction, decimals=2)

    def as_list(self):
        x, y, _ = self.coords
        dx, dy = self.direction
        return [x, y, dx, dy]


class GridShowCase:
    def __init__(self, num_elements, element_dimensions, spacing=[1, 1], space_ratio=[16, 9]):
        self.grid_dimensions = calculate_grid_dimensions(num_elements, ratio=space_ratio)
        self.element_width, self.element_height = element_dimensions
        self.x_spacing, self.y_spacing = spacing

    def get_global_camera_settings(self):
        cols, rows = self.grid_dimensions
        width = cols * self.element_width + (cols - 1) * self.x_spacing
        height = rows * self.element_height + (rows - 1) * self.y_spacing
        size = [width, height]
        position = np.array(size) / 2
        return position, size

    def get_zoomed_camera_settings(self, index):
        bottom_left_x, bottom_left_y = self.get_element_coords(index)
        position = (bottom_left_x + self.element_width/2, bottom_left_y + self.element_height/2)
        size = [self.element_width, self.element_height]
        return position, size

    def get_element_coords(self, index):
        cols, rows = self.grid_dimensions
        x = index % cols
        y = np.floor(index / cols)
        bottom_left_x = x * (self.element_width + self.x_spacing)
        bottom_left_y = y * (self.element_height + self.y_spacing)
        return bottom_left_x, bottom_left_y


def calculate_grid_dimensions(num_elements, ratio):
    factor = (ratio[0] * ratio[1])
    scale = math.sqrt(num_elements / factor)
    grid_dims = [np.ceil(ratio[0] * scale), np.ceil(ratio[1] * scale)]
    approx = ratio[0] * scale * ratio[1] * scale
    # print("Arrange {} elements in ratio {} x {}!".format(num_elements, *ratio))
    # print("scaling is {scale:.2f}! Because {x_scaled:.2f} x {y_scaled:.2f} = {approx:.2f} ~ {num}".format(scale=scale, x_scaled=scale * ratio[0],
    #                                                                                                       y_scaled=scale * ratio[1], num=num_elements,
    #                                                                                                       approx=approx))
    print("Approximate grid dimensions: {} x {}".format(*grid_dims))
    return grid_dims


def draw_graph(graph):
    animation_sequence = []
    node_drawables = [FadeIn(node.drawable) for node in graph.nodes]
    edge_drawables = [Create(edge.drawable) for edge in graph.edges]
    animation_sequence.append(AnimationObject(type='play', content=node_drawables, duration=1, bring_to_front=True))
    animation_sequence.append(AnimationObject(type='play', content=edge_drawables, duration=1, bring_to_back=True))
    return animation_sequence


def add_graph(graph):
    nodes = [node.drawable for node in graph.nodes]
    edges = [edge.drawable for edge in graph.edges]
    return [AnimationObject(type='add', content=nodes),
            AnimationObject(type='add', content=edges, bring_to_back=True)]


def remove_graph(graph, animate=False, duration=1):
    drawables = [node.drawable for node in graph.nodes] + [edge.drawable for edge in graph.edges]
    if animate:
        animations = [FadeOut(drawable) for drawable in drawables]
        return [AnimationObject(type='play', content=animations, duration=duration)]
    return [AnimationObject(type='remove', content=drawables)]


def make_unitary(graph):
    animation_sequence = []

    drawables = graph.remove_all_but_unitary()
    animations = [FadeOut(drawable) for drawable in drawables]
    animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, bring_to_back=False))
    return animation_sequence


def get_circle(coords, radius, color, secondary_color, border_width=2):
    circle = Dot(point=coords, radius=radius)
    circle.set_fill(color, opacity=1)
    circle.set_stroke(secondary_color, width=border_width)
    return circle


def get_line(coord1, coord2, stroke_width=1.0, color=WHITE):
    line = Line(coord1, coord2, stroke_width=stroke_width)
    line.set_color(color)
    return line


def get_square(coords, size, color, secondary_color, border_width=2):
    square = Square(side_length=size)
    square.set_fill(color, opacity=1)
    square.set_stroke(secondary_color, width=border_width)
    square.set_x(coords[0])
    square.set_y(coords[1])
    return square


def get_text(text, coords, scale=1):
    x, y = coords
    text = Tex(text).scale(scale)
    text.set_x(coords[0])
    text.set_y(coords[1])
    return text


def print_2d(grid, print_zeros=True):
    height = len(grid[0])
    width = len(grid)
    for y in range(height - 1, -1, -1):
        row = ""
        for x in range(width):
            if not print_zeros:
                row += "{} ".format(" " if grid[x][y] == 0 else grid[x][y])
            else:
                row += "{} ".format(grid[x][y])
        print(row)
    print()


def get_adjacent(grid, coords):
    """
    Get list of booleans indicating which adjacent cells given coords have.
    :return: list of booleans: [right, up, left, down]
    """
    x, y = coords
    adjacent = []
    for _x, _y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        if x + _x >= len(grid) or y + _y >= len(grid[0]) or x + _x < 0 or y + _y < 0:
            adjacent.append(False)
        else:
            adjacent.append(grid[x + _x][y + _y] > 0)

    return adjacent


if __name__ == '__main__':
    num = 15
    gs = GridShowCase(num, element_dimensions=[4 * 1.3, 4 * 1.3], spacing=[1, 1], space_ratio=[16, 9])
    for i in range(num):
        gs.get_element_coords(i)
