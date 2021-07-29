from ip.ip_util import get_grid_indices, QuantityConstraint, ConditionTypes
from ip.iteration import get_custom_solution, get_imitation_solution, convert_solution_to_graph
from util import TrackProperties
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
        self.root = self.build_feature_model()

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

    def export(self, path):
        tree = ET.ElementTree(self.root.get_xml())
        tree.write(path)


def get_basic_features(graph):
    features = []
    node_grid = graph.grid
    indices = get_grid_indices(len(node_grid), len(node_grid[0]))
    for index, (x, y) in enumerate(indices):
        track_property = node_grid[x][y].track_property
        if track_property is None:
            features.append(StraightStreet("straight", (x, y), suffix=index))
        if track_property in [TrackProperties.turn_90, TrackProperties.turn_180]:
            features.append(CurvedStreet("curve", (x, y), suffix=index))
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
            features.append(Intersection('intersection', coords_list, suffix="i{}".format(len(features))))
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
                features.append(Straight('straight', coords_list, suffix="s{}".format(len(features) + 1)))
            if top > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x + i, y + 1))
                features.append(Straight('straight', coords_list, suffix="s{}".format(len(features) + 1)))
            left, right = vertical_straights[x][y]
            if left > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x, y + i))
                features.append(Straight('straight', coords_list, suffix="s{}".format(len(features) + 1)))
            if right > 0:
                coords_list = []
                for i in range(straight_length):
                    coords_list.append((x + 1, y + i))
                features.append(Straight('straight', coords_list, suffix="s{}".format(len(features) + 1)))
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
