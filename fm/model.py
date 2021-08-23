import pickle
from fm.enums import TLFeatures
from ip.ip_util import get_grid_indices, QuantityConstraint, ConditionTypes, QuantityConstraintStraight
from ip.iteration import get_custom_solution, get_imitation_solution, convert_solution_to_graph
from util import TrackProperties, GraphTour, get_track_points, get_intersection_track_points
from fm.features import Intersection, Straight, StraightStreet, CurvedStreet, Feature, IntersectionConnector
import xml.etree.ElementTree as ET
import numpy as np


class FeatureModel:
    def __init__(self, ip_solution, problem_dict=None, straight_length=3, intersection_size=0.5, scale=1, shift=[0, 0]):
        self.scale = scale
        self.intersection_size = intersection_size
        self.ip_solution = ip_solution
        self.straight_length = straight_length
        if problem_dict is None:
            self.problem_dict = calculate_problem_dict(ip_solution)
        else:
            self.problem_dict = problem_dict

        # convert solution to graph
        self.graph = convert_solution_to_graph(self.ip_solution, self.problem_dict, shift=shift)

        # Extract features from ip solution
        self.features = self.get_features()

        # Scale elements
        self._scale(scale)

        # build feature model and keep reference to root
        self.feature_root = self.build_feature_model()

        # build map
        self.name_to_feature_map = self.build_feature_map()

    def get_features(self):
        intersection_features, intersection_callback = get_intersection_features(self.ip_solution)
        straight_features, coordinates_to_straights = get_straight_features(self.ip_solution, self.problem_dict)
        basic_features = get_basic_features(self.graph, intersection_callback, coordinates_to_straights)
        features = basic_features + intersection_features
        return features

    def build_feature_model(self):
        root = Feature('root', sub_features=self.features, mandatory=True, alternative=False)
        return root

    def build_feature_map(self):
        name_to_feature_map = dict()
        key_value_pairs = self.feature_root.get_mapping()
        for (name, feature) in key_value_pairs:
            name_to_feature_map[name] = feature
        return name_to_feature_map

    def export(self, path):
        # FeatureIDE graphics
        graphics = get_featureIDE_graphics_properties()

        # Structure of Feature Model
        structure = ET.Element('struct')
        structure.append(self.feature_root.get_xml())

        # Constraints
        constraints = ET.Element('constraints')

        # Put it all together
        source_root = ET.Element('extendedFeatureModel')
        source_root.append(graphics)
        source_root.append(structure)
        source_root.append(constraints)

        # write to specified path
        tree = ET.ElementTree(source_root)
        tree.write(path)

    def load_config(self, path):
        with open(path) as f:
            selections = f.readlines()
            for name in selections:
                # print('String clensing:')
                # print(name)
                name = name.replace('"', '')
                name = name.replace('\n', '')
                # print("{}\n\n".format(name))
                self.name_to_feature_map[name].value = True

        for feature in self.features:
            selection = feature.get_selected_sub_features()
            print(f"{feature.type}: {selection}")

    def save(self, path):
        export = {
            'features': self.features,
            'intersection_size': self.intersection_size
        }
        with open(path, 'wb') as handle:
            pickle.dump(export, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _scale(self, factor):
        for feature in self.features:
            feature.scale(factor)
        self.intersection_size *= factor


def get_featureIDE_graphics_properties():
    properties = ET.Element('properties')
    ET.SubElement(properties, 'graphics', key="legendautolayout", value="true")
    ET.SubElement(properties, 'graphics', key="showshortnames", value="false")
    ET.SubElement(properties, 'graphics', key="layout", value="horizontal")
    ET.SubElement(properties, 'graphics', key="showcollapsedconstraints", value="true")
    ET.SubElement(properties, 'graphics', key="legendhidden", value="false")
    ET.SubElement(properties, 'graphics', key="layoutalgorithm", value="1")
    return properties


def get_basic_features(graph, intersection_callback, coordinates_to_straights, intersection_size=0.5):
    features = []
    graph_tour = GraphTour(graph)
    nodes = graph_tour.get_nodes()
    nodes.append(nodes[0])
    nodes.append(nodes[1])
    prev_track_point = None
    prev_track_property = None
    intersection_counter = 0

    for idx in range(len(nodes)-1):
        node1 = nodes[idx]
        node2 = nodes[idx + 1]
        coord1 = node1.get_real_coords()
        coord2 = node2.get_real_coords()
        _, _, track_point = get_track_points(coord1, coord2, 0)
        track_property = node1.track_property
        next_track_property = node2.track_property
        if prev_track_point is None:
            prev_track_point = track_point
            prev_track_property = track_property
            continue

        if track_property is None:
            features.append(StraightStreet(TLFeatures.default.value, track_property, node1.get_coords(), start=prev_track_point, end=track_point, suffix=idx))
            intersection_counter = 0
        elif track_property in [TrackProperties.turn_90, TrackProperties.turn_180]:
            features.append(CurvedStreet(TLFeatures.turn.value, track_property, node1.get_coords(), start=prev_track_point, end=track_point, suffix=idx))
            intersection_counter = 0
        elif track_property is TrackProperties.intersection:
            """
            Intersections handling is special. There are three cases:
            A. We are looking at a a node going into an intersection
            B. We are looking at a node coming from an intersection
            C. We are looking at a node between to intersections
            Which case we look at is determined as follows:
            We know that entries and exits alternate, therefore we keept track with a counter
            If the current node belongs to exactly one intersection (one callback):
            1. If the counter is even, we are in case A
            2. If the counter is odd, we are in case B
            If the current node belongs to two intersection (two callbacks):
            3. We must be in case C
            Note that in case C we must increment the counter by 2, because we are exiting and entering at the same time.
            """
            callbacks = intersection_callback[node1.get_coords()]
            if len(callbacks) == 2:
                track_point1, track_point2 = get_intersection_track_points(prev_track_point, track_point, intersection_size, entering=True, exiting=True)
                feature = IntersectionConnector(TLFeatures.turn.value, TrackProperties.intersection_connector, node1.get_coords(),
                                                start=track_point1, end=track_point2, suffix=idx, entering=True, exiting=True)
                intersection1 = callbacks[0](track_point1, track_point2)
                intersection2 = callbacks[1](track_point1, track_point2)
                if np.linalg.norm(track_point1.logical_coords() - intersection1.center) < np.linalg.norm(track_point2.logical_coords() - intersection1.center):
                    intersection1.add_successor(feature)
                    feature.add_predecessor(intersection1)
                    feature.add_successor(intersection2)
                    intersection2.add_predecessor(feature)
                else:
                    intersection2.add_successor(feature)
                    feature.add_predecessor(intersection2)
                    feature.add_successor(intersection1)
                    intersection1.add_predecessor(feature)
                intersection_counter += 2
            elif len(callbacks) == 1:
                entering = intersection_counter % 2 == 0
                exiting = intersection_counter % 2 == 1
                track_point1, track_point2 = get_intersection_track_points(prev_track_point, track_point, intersection_size, entering=entering, exiting=exiting)
                feature = IntersectionConnector(TLFeatures.turn.value, TrackProperties.intersection_connector, node1.get_coords(),
                                                start=track_point1, end=track_point2, suffix=idx, entering=entering, exiting=exiting)
                intersection = callbacks[0](track_point1, track_point2)
                if entering:
                    intersection.add_predecessor(feature)
                    feature.add_successor(intersection)
                elif exiting:
                    intersection.add_successor(feature)
                    feature.add_predecessor(intersection)
                else:
                    raise RuntimeError("Feature seemingly not connected to intersection")
                intersection_counter += 1
            else:
                raise RuntimeError("No callbacks for intersection found!")

            features.append(feature)

            for callback in callbacks:
                callback(track_point1, track_point2)

        elif track_property is TrackProperties.straight:
            """
            Straights are handled similar to simple straights, except they are only added once.
            Additionally, start and end are set when the corresponding single piece is reached.
            """
            straight = coordinates_to_straights[node1.get_coords()]
            if straight not in features:
                features.append(straight)
            if prev_track_property is not TrackProperties.straight:
                straight.start = prev_track_point
            if next_track_property is not TrackProperties.straight:
                straight.end = track_point

        prev_track_point = track_point
        prev_track_property = track_property

    # link up features
    for index in range(len(features)):
        prev_feature, feature = (features[(index-1) % len(features)], features[index % len(features)])
        if prev_feature.track_property is TrackProperties.intersection_connector and prev_feature.entering:
            # Intersection connectors are connected to their respective intersections, not themselves
            continue
        prev_feature.add_successor(feature)
        feature.add_predecessor(prev_feature)

    return features


def get_intersection_features(ip_solution):
    """
    Create features for intersections.
    Also creates a callback, with which the geometry of the intersections can be filled in based on coordinates.
    Callback: coordinates of element connecting to intersection -> intersection.set_connecting_element()
    """
    intersection_callback = dict()

    features = []
    indices = get_grid_indices(len(ip_solution), len(ip_solution[0]))
    for (x, y) in indices:
        if ip_solution[x][y] == 2:
            coords_list = []
            for (_x, _y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                coords_list.append((x + _x, y + _y))
            intersection = Intersection(TLFeatures.intersection.value, coords_list, suffix="i{}".format(len(features)))
            for (_x, _y, callback) in [(0, 0, intersection.set_bottom_left), (1, 0, intersection.set_bottom_right), (0, 1, intersection.set_top_left), (1, 1, intersection.set_top_right)]:
                intersection_callback.setdefault((x + _x, y + _y), []).append(callback)
            features.append(intersection)
    return features, intersection_callback


def get_straight_features(ip_solution, problem_dict):
    """
    Create features for straights
    """
    width, height = (len(ip_solution), len(ip_solution[0]))
    features = []
    coordinates_to_feature = {}

    if 'horizontal_straights' in problem_dict.keys():
        horizontal_straights = problem_dict['horizontal_straights']
        vertical_straights = problem_dict['vertical_straights']
        for (x, y) in get_grid_indices(width, height):
            for straight_length in range(2, width):
                if straight_length not in horizontal_straights[x][y].keys():
                    continue

                bottom, top = horizontal_straights[x][y][straight_length]
                if bottom > 0:
                    coords_list = []
                    for i in range(straight_length):
                        coords_list.append((x + i, y))
                    feature = Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1))
                    features.append(feature)
                    for i in range(straight_length):
                        coordinates_to_feature[(x + i, y)] = feature
                if top > 0:
                    coords_list = []
                    for i in range(straight_length):
                        coords_list.append((x + i, y + 1))
                    feature = Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1))
                    features.append(feature)
                    for i in range(straight_length):
                        coordinates_to_feature[(x + i, y + 1)] = feature
                left, right = vertical_straights[x][y][straight_length]
                if left > 0:
                    coords_list = []
                    for i in range(straight_length):
                        coords_list.append((x, y + i))
                    feature = Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1))
                    features.append(feature)
                    for i in range(straight_length):
                        coordinates_to_feature[(x, y + i)] = feature
                if right > 0:
                    coords_list = []
                    for i in range(straight_length):
                        coords_list.append((x + 1, y + i))
                    feature = Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1))
                    features.append(feature)
                    for i in range(straight_length):
                        coordinates_to_feature[(x + 1, y + i)] = feature
    return features, coordinates_to_feature


def calculate_problem_dict(ip_solution, print_stats=True):
    """
    Recreate solution with quantity constraints enabled to gather problem dict
    """
    _, problem_dict = get_imitation_solution(ip_solution, print_stats=print_stats)
    return problem_dict


def make_concrete_track():
    quantitiy_constraints = [
        QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, 0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=2, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=3, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=4, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=5, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=6, quantity=0),
    ]
    # Solve Instance with desired properties
    solution, _ = get_custom_solution(6, 6, quantity_constraints=[
        quantitiy_constraints
    ])

    # Create Feature Model
    feature_model = FeatureModel(solution)
    feature_model.export('fm.xml')

    # Use the generated model xml to create a config in FeatureIDE and load this config
    feature_model.load_config('/home/malte/PycharmProjects/circuit-creator/fm/00001.config')

    # Save Feature Model as pickle
    feature_model.save('fm.pkl')


if __name__ == '__main__':
    _solution, _ = get_custom_solution(8, 8, print_stats=False, quantity_constraints=[
        QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, 5),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=2, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=3, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=4, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=5, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=6, quantity=0),
    ])
    fm = FeatureModel(_solution, scale=3)
    fm.save('fm.pkl')
