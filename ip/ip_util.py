from enum import Enum, auto
from pulp import *
import numpy as np
from termcolor import colored


def print_grid(grid, values=True, print_zeros=True, binary=False):
    grid_str = ""
    if values and value(grid[2][2]) is None:
        print("Values None, switching to names...")
        values = False
    for y in range(len(grid[0]) - 1, -1, -1):
        row = ""
        for x in range(len(grid)):
            if values and binary:
                if int(value(grid[x][y])) > 0:
                    print_x_y = colored(str(grid[x][y]), 'green')
                else:
                    print_x_y = grid[x][y]
            elif values:
                print_x_y = "{:<2}".format(int(value(grid[x][y])))
            else:
                print_x_y = grid[x][y]
            if not print_zeros:
                print_x_y = " " if print_x_y == 0 else print_x_y
            row += "{} ".format(print_x_y)
        grid_str += row + '\n'
    grid_str += '\n'
    return grid_str


def print_dict(_dict, values=False, binary=False):
    if values:
        for key in _dict.keys():
            if value(_dict[key][0]) is None:
                print("Values None, switching to names...")
                values = False
                break
    _str = ""
    for key in _dict.keys():
        _list = _dict[key]
        list_str = "{}: ".format(key)
        for variable in _list:
            _print = variable_to_string(variable, values, binary)
            list_str += "{} ".format(_print)
        _str += "{}\n".format(list_str)
    return _str


def print_list(_list, values=False, binary=False, only_positives=False):
    list_str = ""
    for variable in _list:
        if only_positives and value(variable) < 1:
            _print = ""
        else:
            _print = variable_to_string(variable, values, binary)
        list_str += "{} ".format(_print)
    return list_str


def variable_to_string(variable, values, binary):
    if values and binary:
        if int(value(variable)) > 0:
            _print = colored(str(variable), 'green')
        else:
            _print = variable
    elif values:
        _print = int(value(variable))
    else:
        _print = variable

    return _print


def export_grid(grid, save_name=False):
    return [[export_variable(grid[x][y], save_name=save_name) for y in range(len(grid[x]))] for x in range(len(grid))]


def export_dict(_dict, save_name=False):
    export = dict()
    for key in _dict.keys():
        _list = _dict[key]
        export[key] = export_list(_list, save_name=save_name)
    return export


def export_list(_list, save_name=False):
    export = []
    for variable in _list:
        export.append(export_variable(variable, save_name=save_name))
    return export


def export_variable(var, save_name=False):
    # get value
    if var is None:
        var_value = 0
    else:
        var_value = int(value(var))

    # return with or without name
    if save_name:
        return var_value, str(var)
    else:
        return var_value


class TrackProperties(Enum):
    turn_90 = auto()
    turn_180 = auto()
    straight = auto()
    intersection = auto()


class ConditionTypes(Enum):
    less_or_equals = auto()
    more_or_equals = auto()
    equals = auto()


class QuantityConstraint:
    def __init__(self, property_type, condition_type, quantity):
        self.property_type = property_type
        self.condition_type = condition_type
        self.quantity = quantity

    def get_condition(self, variables):
        if self.condition_type == ConditionTypes.less_or_equals:
            return sum(variables) <= self.quantity
        elif self.condition_type == ConditionTypes.more_or_equals:
            return sum(variables) >= self.quantity
        elif self.condition_type == ConditionTypes.equals:
            return sum(variables) == self.quantity


def get_degree_matrix(matrix, value_at_none=0, multipliers=None):
    """
    Given a matrix where each entry is either 1 or 0, return a matrix of equal size where each entry describes how many of the bordering cells are 1
    :param value_at_none: Defines value of degree matrix, where there is no graph. If None, degree is still calculated. Otherwise only positive
                          entries are given a degree and none entries are set as value_at_none
    """
    degree_matrix = np.zeros(np.shape(matrix), dtype=int)
    shift_left = matrix[1:]
    shift_left.append([0] * len(matrix[0]))
    shift_right = matrix[:-1]
    shift_right.insert(0, [0] * len(matrix[0]))
    shift_up = np.array(matrix, dtype=int)[:, :-1]
    shift_up = np.concatenate([np.zeros([len(matrix), 1], dtype=int), shift_up], axis=1)
    shift_down = np.array(matrix, dtype=int)[:, 1:]
    shift_down = np.concatenate([shift_down, np.zeros([len(matrix), 1], dtype=int)], axis=1)

    if multipliers is None:
        degree_matrix += np.array(shift_left) + np.array(shift_right) + np.array(shift_down) + np.array(shift_up)
    else:
        ml, mr, md, mu = multipliers
        degree_matrix += np.array(shift_left) * ml + np.array(shift_right) * mr + np.array(shift_down) * md + np.array(shift_up) * mu

    if value_at_none is not None:
        degree_matrix = np.where(np.array(matrix) > 0, degree_matrix, np.ones_like(degree_matrix) * value_at_none)
    return degree_matrix


def get_intersect_matrix(ip_solution, allow_intersect_at_stubs=False):
    direction_matrix = np.absolute(get_degree_matrix(ip_solution, multipliers=[1, 1, -1, -1]))
    intersect_matrix = np.where(direction_matrix > 1, np.ones_like(ip_solution), np.zeros_like(ip_solution))
    if allow_intersect_at_stubs:
        degree_matrix = get_degree_matrix(ip_solution)
        intersect_matrix = np.where(degree_matrix == 1, np.ones_like(ip_solution), intersect_matrix)

    return intersect_matrix, np.count_nonzero(intersect_matrix)
