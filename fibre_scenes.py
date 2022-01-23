import math
import os

import numpy as np
from manim import *
from termcolor import colored
from tqdm import tqdm

from anim_sequence import AnimationSequenceScene, AnimationObject
from config_parser import Config
from graph import Graph, custom_joins
from ip.ip_util import QuantityConstraint, ConditionTypes, QuantityConstraintStraight, parse_ip_config
from ip.iteration import GraphModel, get_custom_solution, convert_solution_to_graph, get_solution_from_config, ZoneDescription, get_zone_solution, IteratorType
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous, get_interpolation_animation_line, get_angle, \
    get_interpolation_animation_line_continuous
from ip.problem import Problem
from util import Grid, draw_graph, remove_graph, make_unitary, add_graph, generate_track_points, draw_ip_solution, TrackProperties, track_properties_to_colors, get_line, \
    extract_graph_tours, ZoneTypes, TrackPoint, get_text, get_orthogonal_vec, calculate_orthogonal_point, TreeShowCase, tree_to_list, get_circle, get_node_viz
import csv


LIGHT_MODE = False


class SimpleTree(AnimationSequenceScene):
    def construct(self):
        if LIGHT_MODE:
            self.camera.background_color = WHITE

        # render settings
        zoomed_out = 6
        zoomed_in = 1.25
        time_at_node = 1
        element_size = [4, 1]
        scale = 0.1
        node_radius = 1 * scale

        # build tree
        forest = Forest(path_to_csv='/home/malte/svenja/gfibre/Export_CableV5.csv')
        # forest = Forest(path_to_csv='/home/malte/svenja/gfibre/debug3.csv')
        root = forest.roots[0]
        calculate_depth(root)
        tree_showcase = TreeShowCase(root, element_dimensions=element_size, spacing=[2, 0.5], flip=False)

        elements = tree_to_list(root)
        print("Number of Nodes in Tree: {}".format(len(elements)))

        anim_sequence = []

        # move cam
        previous_element = None
        for index, element in tqdm(enumerate(elements)):
            camera_position, camera_size = tree_showcase.get_zoomed_camera_settings(element)
            # print(colored(element, 'green'))
            # print(colored(camera_position, 'green'))

            if previous_element is None:
                # draw node
                anim_sequence.append(AnimationObject(type='add', content=get_node_viz(camera_position, element.id, scale, LIGHT_MODE)))
            else:
                if element.parent == previous_element:
                    previous_position, _ = tree_showcase.get_zoomed_camera_settings(previous_element)
                    line = get_line(previous_position, camera_position, stroke_width=4, color=BLACK if LIGHT_MODE else WHITE)
                    anim_sequence.append(AnimationObject('play', content=Create(line), duration=0.5, z_index=-5))
                    # anim_sequence.append(AnimationObject(type='add', content=get_circle(camera_position, node_radius, BLUE, BLUE_E)))
                    anim_sequence.append(AnimationObject(type='add', content=get_node_viz(camera_position, element.id, scale, LIGHT_MODE)))
                else:
                    previous_position, _ = tree_showcase.get_zoomed_camera_settings(previous_element)
                    parent_position, _ = tree_showcase.get_zoomed_camera_settings(element.parent)
                    line = get_line(parent_position, camera_position, stroke_width=4, color=BLACK if LIGHT_MODE else WHITE)
                    anim_sequence.append(AnimationObject('play', content=Create(line), duration=0.5, z_index=-5))
                    anim_sequence.append(AnimationObject(type='add', content=get_node_viz(camera_position, element.id, scale, LIGHT_MODE)))

            # if index > 15:
            #     break

            previous_element = element

        camera_position, camera_size = tree_showcase.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.5, border_scale=1, shift=[2, -1])
        for anim in anim_sequence:
            self.play_animation(anim)
        self.wait(6)


class MultiLines(AnimationSequenceScene):
    def construct(self):
        # general settings
        colors = [BLUE, GREEN, YELLOW_E, RED, TEAL, PURPLE, GREY_BROWN, ORANGE, DARK_GREY] * 10
        stroke_width = 8
        if LIGHT_MODE:
            self.camera.background_color = WHITE

        # init split
        container = MultiLineSplit(line_splits=[
            LineSplit(num=8, start=TrackPoint((0, 0), (0, 1)), end=TrackPoint((0, 6), (0, 1)), stroke_width=stroke_width, start_spacing=0.01),
            LineSplit(num=8, start=TrackPoint((0, 6), (1, 1)), end=TrackPoint((6, 12), (1, 1)), stroke_width=stroke_width, start_spacing=0.01),
            LineSplit(num=8, start=TrackPoint((6, 12), (1, 0)), end=TrackPoint((12, 12), (1, 0)), stroke_width=stroke_width, start_spacing=0.01)
        ])
        # split = LineSplit(num=8, start=TrackPoint((0, 0), (0, 1)), end=TrackPoint((0, 6), (0, 1)), stroke_width=stroke_width)
        # split = LineSplit(num=8, start=TrackPoint((0, 0), (1, 0)), end=TrackPoint((6, 0), (1, 0)), stroke_width=stroke_width)
        # self.move_camera(*split.get_camera_settings())

        # move cam
        self.move_camera(*container.get_camera_settings())
        animations_list = []

        for split in container.line_splits:
            # animate lines
            for index, line_desc in enumerate(split.create_line_descriptions()):
                interpolation_animation = get_interpolation_animation_line(line_desc, [colors[index]] * 4, stroke_width=stroke_width)
                animations_list.append(interpolation_animation)

            # animate text
            add_text = AnimationObject(type='play', content=split.create_text_descriptions(text_size=0.5, text_color=BLACK if LIGHT_MODE else WHITE))
            animations_list.append([add_text])

        # render
        for anim in animations_list:
            self.play_animations(anim)

        self.wait(4)


class LineSplit:
    def __init__(self, num, start, end, colors=None, stroke_width=4, start_spacing=0.05, split_spacing=0.2, split_percentage=0.2):
        self.num = num
        self.start = start
        self.end = end
        self.distance = np.linalg.norm(np.abs(self.start.logical_coords() - self.end.logical_coords()))
        if colors is None:
            colors = [WHITE] * num
        self.stroke_width = stroke_width
        self.start_spacing = start_spacing
        self.split_spacing = split_spacing
        self.split_percentage = split_percentage

    def create_line_descriptions(self):
        line_descriptions = []
        for position in self.get_split_positions():
            start_point = self.get_orthogonal_point(distance=position * self.start_spacing, percentage=0)
            start_split = self.get_orthogonal_point(distance=position * self.split_spacing, percentage=self.split_percentage)
            end_split = self.get_orthogonal_point(distance=position * self.split_spacing, percentage=1 - self.split_percentage)
            end_point = self.get_orthogonal_point(distance=position * self.start_spacing, percentage=1)
            desc = [start_point, start_split, end_split, end_point]
            line_descriptions.append(desc)
        return line_descriptions

    def create_text_descriptions(self, text_size=0.5, text_color=WHITE, stroke_width=1.5):
        text_list = []
        descriptions = ["description {}".format(index) for index in range(self.num)]
        positions = self.get_split_positions()
        for index, description in enumerate(descriptions):
            rotation = get_angle((1, 0), np.array(self.start.direction[:2]))
            orthogonal = self.get_orthogonal_point(distance=positions[index] * self.split_spacing, percentage=0.5)
            text = get_text(description, orthogonal.logical_coords(), rotate=rotation, scale=text_size, color=text_color,
                            stroke_width=stroke_width)
            text_list.append(text)
        return text_list

    def get_orthogonal_point(self, distance, percentage):
        """
        return track point orthogonal to line from start to end.
        @param distance: distance from line
        @param percentage: position on line in percentage from start to end
        """
        coords_on_line = self.start.logical_coords() + np.array(self.end.logical_coords() - self.start.logical_coords()) * percentage
        point_on_line = TrackPoint(coords_on_line, self.start.direction)
        orthogonal = calculate_orthogonal_point(point_on_line, distance)
        return orthogonal

    def get_split_positions(self):
        indices = [index * 2 for index in range(self.num)]
        median_index = int(indices[-1] / 2)
        positions = np.array(indices) - median_index
        return positions

    def get_camera_settings(self):
        # camera_size = np.abs(self.start.logical_coords() - self.end.logical_coords())
        positions = self.get_split_positions()
        # camera_size = np.abs(self.get_orthogonal_point(self.split_spacing * positions[0], 0).logical_coords() -
        #                      self.get_orthogonal_point(self.split_spacing * positions[-1], 1).logical_coords())
        camera_size = np.abs(self.get_orthogonal_point(self.split_spacing * positions[-1], 0).logical_coords() -
                             self.get_orthogonal_point(self.split_spacing * positions[0], 1).logical_coords())
        camera_pos = np.mean([self.start.logical_coords(), self.end.logical_coords()], axis=0)
        # print("size: {}, camera_pos: {}".format(camera_size, camera_pos))
        return camera_size, camera_pos


class MultiLineSplit:

    def __init__(self, line_splits=None):
        if line_splits is None:
            self.line_splits = []
        else:
            self.line_splits = line_splits

    def add_line_split(self, line_split):
        self.line_splits.append(line_split)

    def get_camera_settings(self):
        min_x, min_y = np.Inf, np.Inf
        max_x, max_y = np.NINF, np.NINF

        for line_split in self.line_splits:
            camera_size, camera_pos = line_split.get_camera_settings()
            camera_size = np.array(camera_size) / 2
            bottom_left = camera_pos - camera_size
            top_right = camera_pos + camera_size
            if bottom_left[0] < min_x:
                min_x = bottom_left[0]
            if bottom_left[1] < min_y:
                min_y = bottom_left[1]
            if top_right[0] > max_x:
                max_x = top_right[0]
            if top_right[1] > max_y:
                max_y = top_right[1]

        bottom_left = np.array([min_x, min_y])
        top_right = np.array([max_x, max_y])

        camera_size = np.abs(top_right - bottom_left)
        camera_pos = np.mean([bottom_left, top_right], axis=0)

        print("size: {}, camera_pos: {}".format(camera_size, camera_pos))
        return camera_size, camera_pos


class Forest:
    def __init__(self, path_to_csv):
        self.roots = []
        # map from node id to Node object
        self.id_to_node = dict()
        # map from node id to Conduit object
        self.id_to_conduit = dict()
        # map from hash(Node, Node) to dict(conduit_id1: list, ..., conduit_idn: list)
        self.node_tuple_to_conduit_list = dict()
        self.parse_csv(path_to_csv)

    def parse_csv(self, path_to_csv):
        current_node = None
        current_conduit = None
        with open(path_to_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                _type, _id, _, _position, _color, _cable, _ = row
                # print("type: {}, id: {}, pos: {}, color: {}, cable: {}".format(_type, _id, _position, _color, _cable))
                if _type == EntryTypes.empty:
                    # print(colored('EMPTY',  'cyan'))
                    current_node = None
                    current_conduit = None
                elif _type == EntryTypes.node:
                    # Check if node is already known
                    if _id in self.id_to_node:
                        # if so, retrieve it from map
                        next_node = self.id_to_node[_id]
                        # special case: node exists and is the root of another tree
                        if next_node.parent is None and current_node is not None:
                            # if so, add tree to this tree
                            next_node.parent = current_node

                    else:
                        # new node
                        next_node = Node(id=_id, depth=0 if current_node is None else current_node.depth + 1, parent=current_node)
                        self.id_to_node[_id] = next_node

                    # If previous node in this chain exists
                    if current_node is not None:
                        # check if the link between current and next node already exists (second conduit)
                        if hash_node_tuple(current_node, next_node) in self.node_tuple_to_conduit_list:
                            # if so, retrieve current list of conduits between two nodes
                            conduit_list = self.node_tuple_to_conduit_list[hash_node_tuple(current_node, next_node)]
                        else:
                            # else make new list and establish link by adding next node as child of current node
                            conduit_list = list()
                            current_node.add_child(next_node)
                        if current_conduit is not None:
                            conduit_list.append(current_conduit)
                        self.node_tuple_to_conduit_list[hash_node_tuple(current_node, next_node)] = conduit_list
                    # else:
                        # new root
                        # self.roots.append(next_node)

                    current_node = next_node
                    current_edges = []
                elif _type == EntryTypes.edge:
                    if current_node is None:
                        raise RuntimeError('Edge without preceding node in csv file!')
                    edge = Edge(_id=_id, _position=_position, _color=_color, _cable=_cable)
                    if current_conduit is None:
                        current_conduit = Conduit(_id)
                        self.id_to_conduit[_id] = current_conduit
                    current_conduit.add_edge(edge)
                else:
                    continue

        for (_, node) in self.id_to_node.items():
            # print("{}".format(str(node)))
            if node.parent is None:
                self.roots.append(node)

        print("Forest has {} root(s)!".format(len(self.roots)))


class Node:
    def __init__(self, id, depth, parent=None):
        self.id = id
        self.depth = depth
        self.parent = parent
        self.children = []
        self.position = None

    def add_child(self, child_node):
        self.children.append(child_node)

    def __str__(self):
        return colored("[NODE] id: {}, depth: {}".format(self.id, self.depth), 'green')


class Edge:
    def __init__(self, _id, _position, _color, _cable):
        self.conduit_id = _id # id of conduit this edge is contained in
        self.position = _position
        self.color = _color
        self.cable = _cable

    def __str__(self):
        return colored("[EDGE] id: {}, cable: {}".format(self.conduit_id, self.cable), 'yellow')


class Conduit:
    def __init__(self, _id):
        self.id = _id
        self.edges = list()

    def add_edge(self, edge):
        if edge.conduit_id != self.id:
            raise ValueError("Trying to add edge to conduit it does not belong to! {} != {}".format(edge.conduit_id, self.id))
        self.edges.append(edge)

    def __str__(self):
        return colored("[CONDUIT] id: {}, num edges: {}".format(self.id, len(self.edges)), 'yellow')


def hash_node_tuple(node1, node2):
    return "{}-{}".format(node1.id, node2.id)


def calculate_depth(root, start_depth=0, dfs=False):
    # calculate and set depth for all nodes in tree
    root.depth = start_depth
    elements = []
    queue = [root]
    while len(queue) > 0:
        # print([str(item) for item in queue])
        if dfs:
            current = queue.pop(len(queue) - 1)
        else:
            current = queue.pop(0)
        elements.append(current)
        for child in current.children:
            child.depth = current.depth + 1
            queue.append(child)

    return elements


class EntryTypes:
    node = 'Node'
    edge = 'ConduitLink'
    empty = 'Empty'


if __name__ == '__main__':
    # scene = MultiLines()
    scene = SimpleTree()
    scene.construct()

    # forest = Forest(path_to_csv='/home/malte/svenja/gfibre/Export_CableV5.csv')
    # forest = Forest(path_to_csv='/home/malte/svenja/gfibre/debug.csv')
