import random
import math
import numpy as np
from manim import *
from anim_sequence import AnimationObject
from io import StringIO
import sys
from enum import Enum, auto
from interpolation import interpolate_single


#######################
##### GRAPH STUFF #####
#######################


class GraphTour:
    """
    This class converts a given graph containing a single cycle to a tour.
    The tour can be retrieved as a sequence of nodes or edges.
    """

    def __init__(self, graph):
        self.graph = graph
        self.nodes = list()
        self.edges = list()
        self.extract_tour()

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

    def get_nodes(self):
        """
        :returns tour as sequence of nodes
        """
        return self.nodes

    def get_edges(self):
        """
        :returns tour as sequence of edges
        """
        return self.edges


def get_next_node(node, previous_node=None):
    """
    Given two nodes, node1 and node2, return any adjacent node to node1 that is not node2.
    Assuming that node1 has degree2, this must return a node leading away from node1 and node2.
    :param node: node1
    :param previous_node: node2
    :returns node that is adjacent to node1 but is not node2
    """
    if node.get_degree() != 2:
        raise ValueError("Graph not 2 factorized!")

    node1, node2 = node.adjacent_nodes

    if previous_node is None or previous_node == node2:
        return node1
    elif previous_node == node1:
        return node2


def get_adjacent(grid, coords):
    """
    Get list of booleans indicating which adjacent cells given coords have in a given grid.
    :returns list of booleans: [right, up, left, down]
    """
    x, y = coords
    adjacent = []
    for _x, _y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        if x + _x >= len(grid) or y + _y >= len(grid[0]) or x + _x < 0 or y + _y < 0:
            adjacent.append(False)
        else:
            adjacent.append(grid[x + _x][y + _y] > 0)

    return adjacent


def is_adjacent(coords1, coords2):
    """
    Check if two coordinates are adjacent, based only on coordinate values.
    :returns True, if manhattan distance between coords1 and coords2 == 1
    TODO: implement manhattan distance instead of for loop
    """
    x1, y1 = coords1
    x2, y2 = coords2
    for _x, _y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        if x1 + _x == x2 and y1 + _y == y2:
            return True
    return False


#######################
##### MANIM STUFF #####
#######################

class Grid:
    """
    Creates a drawable grid, consisting of manim lines
    """

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

    def get_animation_sequence(self, fade_in=True, z_index=-1):
        animation_sequence = []
        if fade_in:
            animations = [FadeIn(drawable) for drawable in self.drawable]
            animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, z_index=z_index))
        else:
            animation_sequence.append(AnimationObject(type='add', content=self.drawable, bring_to_back=True, z_index=z_index))

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


def get_arrow(coord1, coord2, scale=1, color=WHITE):
    arrow = Arrow(coord1, coord2)
    arrow.set_color(color)
    return arrow


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


def draw_graph(graph, z_index=None):
    """
    Creates manim animations for drawing a given graph.
    :returns animations
    """
    animation_sequence = []
    node_drawables = [FadeIn(node.drawable) for node in graph.nodes]
    edge_drawables = [Create(edge.drawable) for edge in graph.edges]
    if z_index is None:
        animation_sequence.append(AnimationObject(type='play', content=node_drawables, duration=1, bring_to_front=True))
        animation_sequence.append(AnimationObject(type='play', content=edge_drawables, duration=1, bring_to_back=True))
    else:
        animation_sequence.append(AnimationObject(type='play', content=node_drawables, duration=1, z_index=z_index+1))
        animation_sequence.append(AnimationObject(type='play', content=edge_drawables, duration=1, bring_to_back=z_index))
    return animation_sequence


def draw_ip_solution(ip_solution, square_size, shift):
    # unpack primary and secondary colors
    pc1, sc1, _ = (GREEN_E, DARK_GREY, 'Positive Cell')
    pc2, sc2, _ = (BLACK, DARK_GREY, 'Negative Cell')

    solution_flat = np.ravel(ip_solution, order='F')
    helper = GridShowCase(len(solution_flat), [square_size, square_size], spacing=[0, 0], space_ratio=[len(ip_solution), len(ip_solution[0])])
    squares = []
    captions = []
    for index in range(len(solution_flat)):
        x, y = (index % len(ip_solution), int(np.floor(index / len(ip_solution))))
        coords = helper.get_element_coords(index)
        if solution_flat[index] > 0:
            square = get_square(np.array(coords) + np.array(shift) + np.array([0.5 * square_size, 0.5 * square_size]), square_size, pc1, sc1,
                                border_width=2 * square_size)
            squares.append(square)
        else:
            square = get_square(coords + np.array(shift) + np.array([0.5 * square_size, 0.5 * square_size]), square_size, pc2, sc2, border_width=2 * square_size)
            squares.append(square)

        captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords + np.array(shift) + np.array([0.5 * square_size, 0.5 * square_size])))

    animation_sequence = [
        AnimationObject('add', content=squares, z_index=0),
        AnimationObject('add', content=captions, z_index=5)
    ]
    return animation_sequence


def add_graph(graph, z_index=0):
    nodes = [node.drawable for node in graph.nodes]
    edges = [edge.drawable for edge in graph.edges]
    return [AnimationObject(type='add', content=edges, z_index=z_index-1), AnimationObject(type='add', content=nodes, z_index=z_index+1)]


def remove_graph(graph, animate=False, duration=1):
    drawables = [node.drawable for node in graph.nodes] + [edge.drawable for edge in graph.edges]
    if animate:
        animations = [FadeOut(drawable) for drawable in drawables]
        return [AnimationObject(type='play', content=animations, duration=duration)]
    return [AnimationObject(type='remove', content=drawables)]


def make_unitary(graph):
    """
    Creates manim animations for removing all but unitary edges of a given graph.
    :returns animations
    """
    animation_sequence = []

    drawables = graph.remove_all_but_unitary()
    animations = [FadeOut(drawable) for drawable in drawables]
    animation_sequence.append(AnimationObject(type='play', content=animations, wait_after=0.5, duration=0.5, bring_to_back=False))
    return animation_sequence


def get_continued_interpolation(track_point, coords, dashed=False, track_color=WHITE, z_index=0):
    """
    Create Animation Object for interpolating a line from a trackpoint to given coordinates, keeping the direction from the trackpoint.
    Used for interpolating the individual lines of an intersection.
    """
    # set direction of track_point to point towards coords
    direction = np.array(coords[:2]) - track_point.coords[:2]
    track_point.direction = direction / np.linalg.norm(direction)
    px, py = interpolate_single(track_point, TrackPoint(coords, track_point.direction))
    line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
    if dashed:
        # choose num dashes in relation to line length
        line = DashedVMobject(line, num_dashes=2, positive_space_ratio=0.6)
    return AnimationObject(type='play', content=[Create(line)], duration=0.25, z_index=z_index)

#######################
#### GEOMETRY STUFF ###
#######################


class TrackPoint:
    """
    A simple point in 2d space, with a direction.
    This class is used for interpolating the track, hence the name track point.
    """

    def __init__(self, coords, direction):
        self.coords = coords
        self.direction = direction / np.linalg.norm(direction)

    def __str__(self):
        return "coords: {}, direction: {}".format(self.coords, self.direction)

    def alter_direction(self, angle):
        """
        Alter direction of point by adding angle (in radian)
        """
        current_angle = np.arctan2(*self.direction)
        new_angle = current_angle + angle
        direction = [np.sin(new_angle), np.cos(new_angle)]
        self.direction = np.round(direction, decimals=2)

    def as_list(self):
        """
        :returns point and direction in one array [x, y, dx, dy]
        """
        x, y = self.coords[:2]
        dx, dy = self.direction
        return [x, y, dx, dy]


def generate_track_points(graph, track_width, z_index=None):
    """
    Generate track points resulting from tour found in given graph.
    For each edge in the tour, three points are generated.
    (in the center, orthogonal to the right of the edge, orthogonal to the left of the edge)
    How far away from the center, left and right are is based on the given track width.
    :return: animations for track points creation, animations for track points removal, track points
    """
    track_points = []
    track_properties = []
    graph_tour = GraphTour(graph)
    nodes = graph_tour.get_nodes()
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
        track_properties.append(node2.track_property)
        line_drawables.append(get_line(center.coords, left.coords, stroke_width=1, color=GREEN))
        line_drawables.append(get_line(center.coords, right.coords, stroke_width=1, color=GREEN))
        point_drawables.append(get_circle(right.coords, 0.04, GREEN, GREEN_E, border_width=1))
        point_drawables.append(get_circle(left.coords, 0.04, GREEN, GREEN_E, border_width=1))

    if z_index is None:
        track_points_creation = [
            AnimationObject(type='play', content=[Create(line) for line in line_drawables], duration=2, bring_to_front=True),
            AnimationObject(type='play', content=[FadeIn(point) for point in point_drawables], duration=1, bring_to_front=True, wait_after=1),
            AnimationObject(type='play', content=[FadeOut(line) for line in line_drawables], duration=0.5)
        ]
    else:
        track_points_creation = [
            AnimationObject(type='play', content=[Create(line) for line in line_drawables], duration=2, z_index=z_index),
            AnimationObject(type='play', content=[FadeIn(point) for point in point_drawables], duration=1, wait_after=1, z_index=z_index+1),
            AnimationObject(type='play', content=[FadeOut(line) for line in line_drawables], duration=0.5)
        ]

    track_points_removal = [
        AnimationObject(type='play', content=[FadeOut(point) for point in point_drawables], duration=1, wait_after=1),
    ]

    return track_points_creation, track_points_removal, track_points, track_properties


def get_track_points(coord1, coord2, track_width):
    """
    :returns right and left point of center of track between two coordinates
    """
    center = find_center(coord1, coord2)
    direction = get_direction(coord1, coord2)
    orth_vec = np.array(get_orthogonal_vec(direction))
    right = np.add(center, track_width * orth_vec)
    left = np.subtract(center, track_width * orth_vec)
    return [TrackPoint(coords, direction) for coords in [right, left, center]]


def get_track_points_from_center(track_point, track_width):
    """
    :returns right and left point orthogonal to coords with direction
    """
    center, direction = (track_point.coords, track_point.direction)
    if len(direction) > 2:
        direction = np.array(direction[:2])
    orth_vec = np.array(get_orthogonal_vec(direction))
    right = np.add(center, track_width * orth_vec)
    left = np.subtract(center, track_width * orth_vec)
    return [TrackPoint(coords, direction) for coords in [right, left, center]]


def alter_track_point_directions(track_points):
    for points in track_points:
        degrees_20 = 0.349066
        angle = degrees_20 * random.uniform(0, 1) - degrees_20 / 2
        [point.alter_direction(angle) for point in points]


def get_intersection_track_points(track_point1, track_point2, entering, exiting):
    """
    Make changes to track points based on their connection to intersections.
    If entering: track_point2 marks the center of an intersection and needs to be moved back
    If exiting: track_point1 marks the center of an intersection and needs to be moved forward
    Note that both entering and exiting can be true, if two intersection are adjacent diagonally.
    """
    # TODO: this doesn't work sadly
    # distance = np.linalg.norm(np.array(track_point2.coords) - np.array(track_point1.coords))
    # size = 0.4 * distance
    size = 0.5
    adjusted1, adjusted2 = (track_point1, track_point2)
    if entering:
        adjusted2 = shift_track_point_along_its_direction(track_point2, -size)
    if exiting:
        adjusted1 = shift_track_point_along_its_direction(track_point1, size)
    return adjusted1, adjusted2


def shift_track_point_along_its_direction(trackpoint, distance):
    direction = trackpoint.direction
    if len(direction) < 3:
        direction = np.array([direction[0], direction[1], 0])
    new_coords = trackpoint.coords + direction * distance
    return TrackPoint(new_coords, trackpoint.direction)


def choose_closest_track_point(origin, track_points):
    min_distance = np.linalg.norm(np.array(track_points[0].coords[:2]) - np.array(origin))
    min_index = 0
    for index, track_point in enumerate(track_points):
        distance = np.linalg.norm(np.array(track_point.coords[:2]) - np.array(origin))
        if distance < min_distance:
            min_distance = distance
            min_index = index
    return track_points[min_index]


def find_center(coord1, coord2):
    x1, y1, _ = coord1
    x2, y2, _ = coord2
    return (x1 + x2) / 2, (y1 + y2) / 2, 0


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    if len(a1) > 2:
        a1 = a1[:2]
    if len(a2) > 2:
        a2 = a2[:2]
    if len(b1) > 2:
        b1 = b1[:2]
    if len(b2) > 2:
        b2 = b2[:2]

    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        raise ValueError("Trying to find intersection of parallel lines")
    return x / z, y / z


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


def transform_coords(coords, shift, scale):
    x, y = coords
    x_shift, y_shift = shift
    return x_shift + x * scale, y_shift + y * scale, 0


#######################
#### GRID SHOWCASE ####
#######################

class GridShowCase:
    """
    A helper class for dividing a 2d space into a grid, and keeping track of the positions of all cells.
    Size and spacing of cells are configurable. The dimensions of the grid are automatically calculated based on a given ratio.
    """

    def __init__(self, num_elements, element_dimensions, spacing=[1, 1], space_ratio=[16, 9], shift=[0, 0]):
        self.grid_dimensions = calculate_grid_dimensions(num_elements, ratio=space_ratio)
        self.element_width, self.element_height = element_dimensions
        self.x_spacing, self.y_spacing = spacing
        self.shift = shift

    def get_global_camera_settings(self):
        """
        :returns position and size of camera, so that all cells will be in view.
        position = center of grid
        size = (width of grid, height of grid)
        """
        cols, rows = self.grid_dimensions
        width = cols * self.element_width + (cols - 1) * self.x_spacing
        height = rows * self.element_height + (rows - 1) * self.y_spacing
        size = [width, height]
        position = np.array(size) / 2
        return self.transform_coords(position), size

    def get_zoomed_camera_settings(self, index):
        """
        :returns position and size of camera, so that a single cell will be in view.
        position = center of cell with given index
        size = (width of cell, height of cell)
        """
        bottom_left_x, bottom_left_y = self.get_element_coords(index)
        position = (bottom_left_x + self.element_width / 2, bottom_left_y + self.element_height / 2)
        size = [self.element_width, self.element_height]
        return self.transform_coords(position), size

    def get_element_coords(self, index):
        """
        :returns position of bottom left corner of cell with given index.
        """
        cols, rows = self.grid_dimensions
        x = index % cols
        y = np.floor(index / cols)
        bottom_left_x = x * (self.element_width + self.x_spacing)
        bottom_left_y = y * (self.element_height + self.y_spacing)
        return self.transform_coords((bottom_left_x, bottom_left_y))

    def transform_coords(self, coords):
        x, y = coords
        x_shift, y_shift = self.shift
        return [x + x_shift, y + y_shift]


def calculate_grid_dimensions(num_elements, ratio):
    """
    Calculates how n elements should be arranged to best fit a given ratio.
    :returns grid dimensions [width, height]
    """
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


#######################
##### BASIC STUFF #####
#######################

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


def add_to_list_in_dict(_dict, key, element):
    if key not in _dict.keys():
        _dict[key] = [element]
    else:
        _dict[key].append(element)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


########################
## ENUMS AND MAPPINGS ##
########################

class TrackProperties(Enum):
    turn_90 = auto()
    turn_180 = auto()
    straight = auto()
    intersection = auto()
    intersection_connector = auto()


def track_properties_to_colors(track_properties):
    colors = []
    property_to_color = {
        TrackProperties.turn_90: PINK,
        TrackProperties.turn_180: BLUE_C,
        TrackProperties.straight: RED,
        TrackProperties.intersection: YELLOW_E,
        TrackProperties.intersection_connector: GREEN
    }
    for track_property in track_properties:
        if track_property is None:
            colors.append(WHITE)
        else:
            colors.append(property_to_color[track_property])
    if len(colors) == 1:
        return colors[0]
    return colors


if __name__ == '__main__':
    # num = 15
    # gs = GridShowCase(num, element_dimensions=[4 * 1.3, 4 * 1.3], spacing=[1, 1], space_ratio=[16, 9])
    # for i in range(num):
    #     gs.get_element_coords(i)
    print(count_adjacent(np.ones((3, 3), dtype=int), (2, 2)))
