import json
import math
import os.path
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from argparse import Namespace
import numpy as np
from termcolor import colored
from tqdm import tqdm
from fm.model import FeatureModel
from ip.ip_util import ConditionTypes, QuantityConstraintStraight
from ip.iteration import ZoneDescription, get_custom_solution, get_imitation_solution, get_zone_assignment, FullProhibitionIterator
from util import TrackProperties, ZoneTypes

# path_to_configs = os.path.join(os.getcwd(), 'super_configs')


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

zone_type_to_name = {
    ZoneTypes.parking: 'parking',
    ZoneTypes.urban_area: 'urban_area',
    ZoneTypes.no_passing: 'no_passing',
    ZoneTypes.express_way: 'express_way',
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
        self.evaluation = parse_features(config_dict.get('evaluation'))

    def iterate_layouts(self, out_path, num=math.inf, _print=False, generate_images=True):
        os.makedirs(os.path.join(out_path, 'fm'), exist_ok=True)
        iterator = FullProhibitionIterator(self, _print=_print)
        solutions = iterator.iterate(num_solutions=num)

        for index, solution in enumerate(solutions):
            print(self.zones.descriptions)
            zone_assignment, start_index = get_zone_assignment(solution, self.zones.descriptions)
            fm = self.get_features(solution, zone_assignment, start_index=start_index)
            fm_path = os.path.join(out_path, 'fm', 'fm{}.pkl'.format(index))
            fm.save(fm_path)

            # write config to new file
            self.layout.solution = solution
            self.zones.solution = zone_assignment
            self.features.fm_path = fm_path
            config_path = os.path.join(out_path, 'config{}.json'.format(index))
            self.write(config_path)

        if generate_images:
            visualize(out_path)

        return len(solutions)

    def get_fm(self):
        ip_solution, problem_dict = self.get_layout()
        zone_assignment, start_index = self.get_zones(ip_solution, problem_dict)
        fm = self.get_features(ip_solution, zone_assignment, problem_dict, start_index)
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
            print(colored("Could not load featureIDE config! Config does not match model", 'red'))
            print(colored("Init fm with no selected features", 'yellow'))
        except FileNotFoundError:
            print(colored("Could not load featureIDE config! Given config does not exist: {}".format(path_to_config), 'red'))
            print(colored("Init fm with no selected features", 'yellow'))

        return fm

    def write(self, path):
        layout_dict = vars(deepcopy(self.layout))
        layout_dict['constraints'] = layout_dict['constraints_raw']
        del layout_dict['constraints_raw']
        zone_dict = vars(deepcopy(self.zones))
        zone_dict['descriptions'] = zone_dict['descriptions_raw']
        zone_dict['solution'] = zone_solution_to_dict(self.zones.solution)
        del zone_dict['descriptions_raw']
        data = {
            'layout': layout_dict,
            'zones': zone_dict,
            'features': vars(deepcopy(self.features)),
            'evaluation': vars(deepcopy(self.evaluation))
        }
        if Path(path).suffix != '.json':
            path += '.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return path


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
    layout_dict['constraints_raw'] = layout_dict.get('constraints')
    layout_dict['constraints'] = constraints

    # Parse solution
    solution = None
    if layout_dict.get('solution') is not None:
        solution = layout_dict['solution']
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

    zone_dict['descriptions_raw'] = zone_dict.get('descriptions')
    zone_dict['descriptions'] = zone_descriptions

    # Parse solution
    solution = None
    if zone_dict.get('solution') is not None:
        solution = dict()
        zone_assignment = zone_dict.get('solution')
        for key in zone_assignment.keys():
            solution[zone_name_to_type[key]] = zone_assignment[key]
    zone_dict['solution'] = solution

    return Namespace(**zone_dict)


def zone_solution_to_dict(zone_solution):
    if zone_solution is None:
        return zone_solution
    # Parse solution
    solution_dict = dict()
    for key in zone_solution.keys():
        solution_dict[zone_type_to_name[key]] = zone_solution[key]
    return solution_dict


def parse_features(fm_dict):
    if fm_dict is None:
        return None
    return Namespace(**fm_dict)


def parse_evaluation(eval_dict):
    if eval_dict is None:
        return None
    return Namespace(**eval_dict)


def generate_configs(blueprint_config, output_path, num=10):
    """
    Generate a number of solution configs that match a given blueprint.
    Write generate configs to files at output_path.
    """
    feature_models = blueprint_config.iterate_layouts()
    print("{} layouts found!".format(len(feature_models)))

    pass


def visualize(path_to_configs, path_to_viz=None, tmp_path='/tmp/config.json'):
    if path_to_viz is None:
        path_to_viz = os.path.join(path_to_configs, 'viz')

    os.makedirs(path_to_viz, exist_ok=True)

    configs = []
    filenames = [f for f in os.listdir(path_to_configs) if os.path.isfile(os.path.join(path_to_configs, f))]
    filenames.sort(key=lambda f: int(Path(f).stem[len('config'):]))
    for index, fname in tqdm(enumerate(filenames), desc='rendering images'):
        if Path(fname).suffix == '.json':
            full_path = os.path.join(path_to_configs, fname)
            print("loading config from {}".format(full_path))
            configs.append(Config(full_path))

            # move config to tmp path
            shutil.copyfile(full_path, tmp_path)

            # render image with system call
            subprocess.call(['python', '-m', 'manim', '-s', 'creation_scenes.py', 'DrawSuperConfig'])

            # move image to correct folder
            manim_location = os.path.join(os.getcwd(), 'media/images/creation_scenes', 'DrawSuperConfig_ManimCE_v0.8.0.png')
            new_location = os.path.join(path_to_viz, 'image{}.png'.format(index))
            print("Copying image from {} to  {}".format(colored(manim_location, 'yellow'), colored(new_location, 'cyan')))
            shutil.copyfile(manim_location, new_location)


if __name__ == '__main__':
    path_to_blueprint = '/home/malte/PycharmProjects/circuit-creator/super_configs/config.json'
    output_path = '/home/malte/PycharmProjects/circuit-creator/super_configs/test'
    config = Config(path_to_blueprint)
    config.iterate_layouts(output_path)
