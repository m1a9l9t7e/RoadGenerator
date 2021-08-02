from fm.enums import TLFeatures
from ip.ip_util import get_grid_indices, QuantityConstraint, ConditionTypes
from ip.iteration import get_custom_solution, get_imitation_solution, convert_solution_to_graph
from util import TrackProperties, GraphTour, get_track_points, get_intersection_track_point
from fm.features import Intersection, Straight, StraightStreet, CurvedStreet, Feature
import xml.etree.ElementTree as ET


class FeatureModel:
    def __init__(self, ip_solution, problem_dict=None, straight_length=3):
        self.ip_solution = ip_solution
        if problem_dict is None:
            self.problem_dict = calculate_problem_dict(ip_solution)
        else:
            self.problem_dict = problem_dict
        self.straight_length = straight_length

        # convert solution to graph
        self.graph = convert_solution_to_graph(self.ip_solution, self.problem_dict, self.straight_length)

        # Extract features from ip solution
        self.features = self.get_features()

        # build feature model and keep reference to root
        self.feature_root = self.build_feature_model()

        # build map
        self.name_to_feature_map = self.build_feature_map()

    def get_features(self):
        basic_features = get_basic_features(self.graph)
        intersection_features = get_intersection_features(self.ip_solution)
        straight_features = get_straight_features(self.ip_solution, self.problem_dict, self.straight_length)
        print("Basic features: {}, intersections: {}, straights: {}".format(len(basic_features), len(intersection_features), len(straight_features)))
        features = basic_features + intersection_features + straight_features
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


def get_featureIDE_graphics_properties():
    properties = ET.Element('properties')
    ET.SubElement(properties, 'graphics', key="legendautolayout", value="true")
    ET.SubElement(properties, 'graphics', key="showshortnames", value="false")
    ET.SubElement(properties, 'graphics', key="layout", value="horizontal")
    ET.SubElement(properties, 'graphics', key="showcollapsedconstraints", value="true")
    ET.SubElement(properties, 'graphics', key="legendhidden", value="false")
    ET.SubElement(properties, 'graphics', key="layoutalgorithm", value="1")
    return properties


def get_basic_features(graph):
    features = []
    graph_tour = GraphTour(graph)
    nodes = graph_tour.get_nodes()
    nodes.append(nodes[0])
    nodes.append(nodes[1])
    prev_track_point = None
    prev_track_property = None

    for idx in range(len(nodes)-1):
        node1 = nodes[idx]
        node2 = nodes[idx + 1]
        coord1 = node1.get_real_coords()
        coord2 = node2.get_real_coords()
        _, _, track_point = get_track_points(coord1, coord2, 0)
        track_property = node1.track_property
        if prev_track_point is None:
            prev_track_point = track_point
            prev_track_property = track_property
            continue

        if track_property is None:
            features.append(StraightStreet(TLFeatures.default.value, track_property, node1.get_coords(), start=prev_track_point, end=track_point, suffix=idx))
        elif track_property in [TrackProperties.turn_90, TrackProperties.turn_180]:
            features.append(CurvedStreet(TLFeatures.turn.value, track_property, node1.get_coords(), start=prev_track_point, end=track_point, suffix=idx))
        elif track_property is TrackProperties.intersection:
            track_point1, track_point2 = get_intersection_track_point(prev_track_point, track_point, entering=prev_track_property is TrackProperties.intersection)
            features.append(CurvedStreet(TLFeatures.turn.value, TrackProperties.intersection_connector, node1.get_coords(), start=track_point1, end=track_point2, suffix=idx))

        prev_track_point = track_point
        prev_track_property = track_property

    return features


def get_intersection_features(ip_solution):
    """
    Create features for intersections
    """
    features = []

    indices = get_grid_indices(len(ip_solution), len(ip_solution[0]))
    for (x, y) in indices:
        if ip_solution[x][y] == 2:
            coords_list = []
            for (_x, _y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                coords_list.append((x + _x, y + _y))
            features.append(Intersection(TLFeatures.intersection.value, coords_list, suffix="i{}".format(len(features))))
    return features


def get_straight_features(ip_solution, problem_dict, straight_length):
    """
    Create features for straights
    """
    width, height = (len(ip_solution), len(ip_solution[0]))
    features = []

    if 'horizontal_straights' in problem_dict.keys():
        horizontal_straights = problem_dict['horizontal_straights']
        vertical_straights = problem_dict['vertical_straights']
        for (x, y) in get_grid_indices(width, height):
            bottom, top = horizontal_straights[x][y]
            if bottom > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x + i, y))
                features.append(Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1)))
            if top > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x + i, y + 1))
                features.append(Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1)))
            left, right = vertical_straights[x][y]
            if left > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x, y + i))
                features.append(Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1)))
            if right > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x + 1, y + i))
                features.append(Straight(TLFeatures.straight.value, coords_list, suffix="s{}".format(len(features) + 1)))
    return features


def calculate_problem_dict(ip_solution):
    """
    Recreate solution with quantity constraints enabled to gather problem dict
    """
    _, problem_dict = get_imitation_solution(ip_solution, print_stats=False)
    return problem_dict


if __name__ == '__main__':
    solution, _ = get_custom_solution(6, 6, quantity_constraints=[
        QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, 1),
        QuantityConstraint(TrackProperties.straight, ConditionTypes.equals, 1)
    ])
    fm = FeatureModel(solution)
    fm.export('fm.xml')
    fm.load_config('/home/malte/PycharmProjects/circuit-creator/fm/00001.config')
