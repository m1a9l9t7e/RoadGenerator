import json
from pprint import pprint

import numpy as np

from ip.ip_util import QuantityConstraint, ConditionTypes
from util import TrackProperties

property_name_to_type = {
    'intersection': TrackProperties.intersection,
    'turn_90': TrackProperties.turn_90,
    'turn_180': TrackProperties.turn_180,
    'straight': TrackProperties.straight,
}


class Config:
    def __init__(self, path=None):
        if path is None:
            config_dict = dict()
        else:
            config_dict = json.load(open(path))

        layout = parse_layout(config_dict.get('layout'))
        zones = parse_zones(config_dict.get('zones'))
        features = parse_features(config_dict.get('features'))
        pprint(layout)


def parse_layout(layout_dict):
    if layout_dict is None:
        return dict()

    # parse constraints
    constraints = []
    if layout_dict.get('constraints') is not None:
        for constraint in layout_dict['constraints']:
            if constraint.get('min') is not None and constraint.get('max') is not None:
                if constraint.get('min') == constraint.get('max'):
                    q_constraint_equals = QuantityConstraint(
                        property_type=property_name_to_type[constraint.get('type')],
                        condition_type=ConditionTypes.equals,
                        quantity=constraint.get('max')
                    )
                    constraints.append(q_constraint_equals)
                    continue
            if constraint.get('min') is not None:
                q_constraint_lower = QuantityConstraint(
                    property_type=property_name_to_type[constraint.get('type')],
                    condition_type=ConditionTypes.more_or_equals,
                    quantity=constraint.get('min')
                )
                constraints.append(q_constraint_lower)
            if constraint.get('max') is not None:
                q_constraint_upper = QuantityConstraint(
                    property_type=property_name_to_type[constraint.get('type')],
                    condition_type=ConditionTypes.less_or_equals,
                    quantity=constraint.get('max')
                )
                constraints.append(q_constraint_upper)
    layout_dict['constraints'] = constraints

    # Parse solution
    solution = None
    if layout_dict.get('solution') is not None:
        solution = np.array(layout_dict['solution'])
    layout_dict['solution'] = solution

    return layout_dict


def parse_zones(layout_dict):
    if layout_dict is None:
        return dict()
    return layout_dict


def parse_features(layout_dict):
    if layout_dict is None:
        return dict()
    return layout_dict


def generate_config():
    pass


if __name__ == '__main__':
    config = Config('/home/malte/PycharmProjects/circuit-creator/super_configs/config.json')
