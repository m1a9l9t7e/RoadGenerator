import json
import math
import os.path
from pathlib import Path
from pprint import pprint
from argparse import Namespace
import numpy as np
from termcolor import colored

from fm.model import FeatureModel
from ip.ip_util import QuantityConstraint, ConditionTypes, QuantityConstraintStraight
from ip.iteration import ZoneDescription, get_custom_solution, get_imitation_solution, get_zone_assignment, FullProhibitionIterator
from util import TrackProperties, ZoneTypes

path_to_configs = os.path.join(os.getcwd(), 'super_configs')


property_name_to_type = {
    'intersection': TrackProperties.intersection,
    'turn_90': TrackProperties.turn_90,
    'turn_180': TrackProperties.turn_180,
    'straight': TrackProperties.straight,
}

zone_name_to_type = {
    'parking': ZoneTypes.parking,
    'urban_area': ZoneTypes.urban_area,
    'no_passing': ZoneTypes.no_passing,
    'express_way': ZoneTypes.express_way,
}


class Config:
    def __init__(self, path=None):
        self.dimensions = None
        if path is None:
            config_dict = dict()
            self.path = 'none'
        else:
            config_dict = json.load(open(path))
            self.path = path

        self.layout = parse_layout(config_dict.get('layout'))
        if self.layout.solution is not None:
            self.dimensions = [value + 1 for value in np.shape(self.layout.solution)]
        else:
            self.dimensions = (self.layout.width, self.layout.height)

        self.zones = parse_zones(config_dict.get('zones'))
        self.features = parse_features(config_dict.get('features'))

    def iterate_layouts(self, num=math.inf, _print=False):
        iterator = FullProhibitionIterator(self, _print=_print)
        solutions = iterator.iterate(num_solutions=num)

        fms = []
        for solution in solutions:
            zone_assignment, start_index = get_zone_assignment(solution, self.zones.descriptions)
            fm, fm_path = self.get_features(solution, zone_assignment, start_index=start_index)
            fms.append(fm)

        return fms

    def get_fm(self):
        ip_solution, problem_dict = self.get_layout()
        zone_assignment, start_index = self.get_zones(ip_solution, problem_dict)
        fm, fm_path = self.get_features(ip_solution, zone_assignment, problem_dict, start_index)
        print(colored("Full fm generated at: {}".format(os.path.abspath(fm_path)), 'green'))
        # TODO generate sdf track with system call?
        return fm

    def get_layout(self):
        layout = self.layout
        if layout.solution is not None:
            ip_solution, problem_dict = get_imitation_solution(layout.solution, print_stats=True)
            self.dimensions = [value + 1 for value in np.shape(ip_solution)]
        else:
            self.dimensions = (layout.width, layout.height)
            ip_solution, problem_dict = get_custom_solution(*self.dimensions, quantity_constraints=layout.constraints, print_stats=True,
                                                            allow_gap_intersections=layout.allow_gap_intersections)
            self.layout.solution = ip_solution
        return ip_solution, problem_dict

    def get_zones(self, ip_solution, problem_dict=None):
        if self.zones.solution is not None:
            zone_assignment = self.zones.solution
            start_index = None  # TODO
        else:
            zone_assignment, start_index = get_zone_assignment(ip_solution, self.zones.descriptions, problem_dict)
            self.zones.solution = zone_assignment
        return zone_assignment, start_index

    def get_features(self, ip_solution, zone_assignment, problem_dict=None, start_index=None):
        fm = FeatureModel(ip_solution, zone_assignment, scale=self.layout.scale, problem_dict=problem_dict, start_index=start_index)
        featureide_source = 'fm-{}.xml'.format(Path(self.path).stem)
        fm.export(featureide_source)

        # TODO: generate featureIDE configs with java call -->
        path_to_config = '/fm/00002.config'
        # TODO: generate featureIDE configs with java call <--

        try:
            fm.load_config(path_to_config)
        except KeyError:
            print(colored("Could not load config! Config does not match model", 'red'))
        except FileNotFoundError:
            print(colored("Could not load config! Given config does not exist: {}".format(path_to_config), 'red'))

        full_save_path = 'fm-{}.pkl'.format(Path(self.path).stem)
        return fm, full_save_path


def parse_layout(layout_dict):
    if layout_dict is None:
        return None

    # parse constraints
    constraints = []
    if layout_dict.get('constraints') is not None:
        for constraint in layout_dict['constraints']:
            if constraint.get('min') is not None and constraint.get('max') is not None:
                if constraint.get('min') == constraint.get('max'):
                    q_constraint_equals = QuantityConstraintStraight(
                        property_type=property_name_to_type[constraint.get('type')],
                        condition_type=ConditionTypes.equals,
                        quantity=constraint.get('max'),
                        length=constraint.get('length')
                    )
                    constraints.append(q_constraint_equals)
                    continue
            if constraint.get('min') is not None:
                q_constraint_lower = QuantityConstraintStraight(
                    property_type=property_name_to_type[constraint.get('type')],
                    condition_type=ConditionTypes.more_or_equals,
                    quantity=constraint.get('min'),
                    length = constraint.get('length')
                )
                constraints.append(q_constraint_lower)
            if constraint.get('max') is not None:
                q_constraint_upper = QuantityConstraintStraight(
                    property_type=property_name_to_type[constraint.get('type')],
                    condition_type=ConditionTypes.less_or_equals,
                    quantity=constraint.get('max'),
                    length=constraint.get('length')
                )
                constraints.append(q_constraint_upper)
    layout_dict['constraints'] = constraints

    # Parse solution
    solution = None
    if layout_dict.get('solution') is not None:
        solution = np.array(layout_dict['solution'])
    layout_dict['solution'] = solution

    # pprint(layout_dict)
    return Namespace(**layout_dict)


def parse_zones(zone_dict):
    if zone_dict is None:
        return None

    # Parse descriptions
    zone_descriptions = []
    if zone_dict.get('descriptions') is not None:
        for description in zone_dict['descriptions']:
            z_description = ZoneDescription(zone_name_to_type[description.get('type')], min_length=description.get('min_length'), max_length=description.get('max_length'))
            zone_descriptions.append(z_description)

    zone_dict['descriptions'] = zone_descriptions

    # Parse solution
    solution = None
    if zone_dict.get('solution') is not None:
        solution = np.array(zone_dict['solution'])
    zone_dict['solution'] = solution

    return Namespace(**zone_dict)


def parse_features(fm_dict):
    if fm_dict is None:
        return None
    return Namespace(**fm_dict)


def generate_config():
    pass


if __name__ == '__main__':
    path_to_config = '/home/malte/PycharmProjects/circuit-creator/super_configs/config.json'
    config = Config(path_to_config)
    feature_models = config.iterate_layouts()
    print("{} layouts found!".format(len(feature_models)))
