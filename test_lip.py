import json
import math
import os.path
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from argparse import Namespace
from pprint import pprint

import numpy as np
from termcolor import colored
from tqdm import tqdm

from config_parser import Config, visualize
from fm.model import FeatureModel
from ip.ip_util import ConditionTypes, QuantityConstraintStraight, QuantityConstraint
from ip.iteration import ZoneDescription, get_custom_solution, get_imitation_solution, get_zone_assignment, FullProhibitionIterator, convert_solution_to_graph, \
    find_straight_zones
from ip.problem import Problem
from util import TrackProperties, ZoneTypes, print_2d, time, extract_graph_tours
import random

out_path = '/tmp/viz'


def verify_lip(num=1000, _print=True, generate_images=False):
    os.makedirs(out_path, exist_ok=True)

    feasible = 0
    infeasible = 0

    for i in range(num):
        # create random constraints
        num_intersections = get_random_limits(0, 4)
        num_turn90 = get_random_limits(0, 12)
        num_turn180 = get_random_limits(0, 12)
        num_s2 = get_random_limits(0, 4)
        num_s3 = get_random_limits(0, 2)
        num_s4 = get_random_limits(0, 2)
        num_s5 = get_random_limits(0, 0)
        num_s6 = get_random_limits(0, 0)
        straight_limits = {2: num_s2, 3: num_s3, 4: num_s4, 5: num_s5, 6:num_s6}

        print("{} <= i <= {}".format(*num_intersections))
        print("{} <= t90 <= {}".format(*num_turn90))
        print("{} <= t180 <= {}".format(*num_turn180))
        print("{} <= s2 <= {}".format(*num_s2))
        print("{} <= s3 <= {}".format(*num_s3))
        print("{} <= s4 <= {}".format(*num_s4))
        print("{} <= s5 <= {}".format(*num_s5))
        print("{} <= s6 <= {}".format(*num_s6))

        # solve problem
        _quantity_constraints = [
            QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, quantity=num_intersections[0]),
            QuantityConstraint(TrackProperties.intersection, ConditionTypes.less_or_equals, quantity=num_intersections[1]),
            QuantityConstraint(TrackProperties.turn_180, ConditionTypes.more_or_equals, quantity=num_turn90[0]),
            QuantityConstraint(TrackProperties.turn_180, ConditionTypes.less_or_equals, quantity=num_turn90[1]),
            QuantityConstraint(TrackProperties.turn_90, ConditionTypes.more_or_equals, quantity=num_turn180[0]),
            QuantityConstraint(TrackProperties.turn_90, ConditionTypes.less_or_equals, quantity=num_turn180[1]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=2, quantity=num_s2[0]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.less_or_equals, length=2, quantity=num_s2[1]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=3, quantity=num_s3[0]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.less_or_equals, length=3, quantity=num_s3[1]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=4, quantity=num_s4[0]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.less_or_equals, length=4, quantity=num_s4[1]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=5, quantity=num_s5[0]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.less_or_equals, length=5, quantity=num_s5[1]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=6, quantity=num_s6[0]),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.less_or_equals, length=6, quantity=num_s6[1]),
        ]

        p = Problem(7, 3, quantity_constraints=_quantity_constraints, allow_gap_intersections=True)

        start = time.time()
        _solution, status = p.solve(_print=False)
        end = time.time()
        print(colored("Problem solved! Time elapsed: {:.2f}s".format(end - start), "green" if status > 1 else "red"))

        if status > 1:
            p.get_stats()
            feasible += 1
        else:
            infeasible += 1
            continue

        problem_dict = p.export_variables()
        graph = convert_solution_to_graph(_solution, problem_dict)
        graph_tours = extract_graph_tours(graph)

        # drive-straight only
        if len(graph_tours) > 1:
            print(colored("Skipping non drive-straight track!", "blue"))
            continue

        # verify correctness of constraints in solution
        # 1. intersections
        lower, upper = num_intersections
        if not (lower <= np.count_nonzero(np.array(_solution) >= 2) <= upper):
            RuntimeError('Intersection requirement not fulfilled!')

        # 2. turns
        t90, t180 = count_turns(graph_tours[0])
        p.get_stats()
        lower, upper = num_turn90
        if not (lower <= t90 <= upper):
            RuntimeError('90 degree turn requirement not fulfilled!')
        lower, upper = num_turn180
        if not (lower <= t180 <= upper):
            RuntimeError('180 degree turn requirement not fulfilled!')

        # 3. straights
        graph_tour = graph_tours[0]
        straight_zones = find_straight_zones(graph_tour)

        straight_tuples = []
        for length in straight_zones.keys():
            lower, upper = straight_limits[length]
            if lower <= len(straight_zones[length]) <= upper:
                continue
            else:
                raise RuntimeError('Straight requirement for l={} not fulfilled!'.format(length))

        print(colored("All requirements fulfilled!", "green"))

        config = Config('/home/malte/PycharmProjects/circuit-creator/super_configs/cc17_debug.json')
        config.layout.solution = _solution

        config_path = os.path.join(out_path, 'config{}.json'.format(i))
        config.write(config_path)

    if generate_images:
        os.makedirs('tmp/pretty_pictures', exist_ok=True)
        visualize(out_path, path_to_viz='tmp/pretty_pictures')

    print(colored("{}/{} configs feasible. All feasible were correct!".format(feasible, feasible+infeasible), "green"))


def count_turns(graph_tour):
    nodes = deepcopy(graph_tour.nodes)
    nodes += nodes[:2]
    previous_direction = None
    current_direction = None
    last_direction_changes = []

    turn_90_counter = 0
    turn_180_counter = 0

    for index in range(len(nodes) - 1):
        current_node = nodes[index]
        next_node = nodes[index + 1]
        current_direction = np.array(next_node.get_coords()) - np.array(current_node.get_coords())
        if previous_direction is not None:
            # if not np.array(current_direction == previous_direction).all():
            #     direction_change_counter += 1
            if np.sum(np.abs(np.array(current_direction - previous_direction))) == 2:
                change = get_direction(previous_direction, current_direction)
                last_direction_changes.append(change)
                # print("turn detected, {} in a row. Direction = {}. Last changes: {}".format(len(last_direction_changes), change, last_direction_changes))
                if len(last_direction_changes) == 2:
                    if last_direction_changes[0] == last_direction_changes[1]:
                        turn_180_counter += 1
                        last_direction_changes = []
                        # print("+1 180")
            else:
                if len(last_direction_changes) == 1:
                    turn_90_counter += 1
                    last_direction_changes = []
                    # print("+1 90")
                elif len(last_direction_changes) == 2:
                    if last_direction_changes[0] == last_direction_changes[1]:
                        turn_180_counter += 1
                        last_direction_changes = []
                        # print("+1 180")
                    else:
                        turn_90_counter += 2
                        last_direction_changes = []
                        # print("+2 90")
            if len(last_direction_changes) > 2:
                turn_90_counter += 1
                last_direction_changes = last_direction_changes[:1]
                # print("+1 90")

        previous_direction = current_direction

    if len(last_direction_changes) == 1:
        turn_90_counter += 1
        # print("+1 90")
    elif len(last_direction_changes) == 2:
        if last_direction_changes[0] == last_direction_changes[1]:
            turn_180_counter += 1
            # print("+1 180")
        else:
            turn_90_counter += 2
            # print("+2 90")

    return turn_90_counter, turn_180_counter


def get_angle(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    angle = np.arccos(dot_product)
    return angle


def get_direction(vec1, vec2):
    if vec1[0] == 1:
        if vec2[1] > 0:
            return 'left'
        else:
            return 'right'
    if vec1[0] == -1:
        if vec2[1] > 0:
            return 'right'
        else:
            return 'left'
    if vec1[1] == 1:
        if vec2[0] > 0:
            return 'right'
        else:
            return 'left'
    if vec1[1] == -1:
        if vec2[0] > 0:
            return 'left'
        else:
            return 'right'


def get_random_limits(_min, _max):
    lower = random.randint(_min, _max)
    upper = random.randint(lower, _max)
    return lower, upper
    # return 0, 1000


if __name__ == '__main__':
    verify_lip(num=1000, generate_images=False)
