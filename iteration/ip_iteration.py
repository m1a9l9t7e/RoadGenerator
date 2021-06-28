from graph import Graph, GraphSearcher
from ip import Problem
from util import draw_graph, make_unitary, GridShowCase
import numpy as np
import itertools
from termcolor import colored


class GraphModel:
    def __init__(self, width, height, generate_intersections=True):
        self.width = width - 1
        self.height = height - 1
        self.variants = iterate(self.width, self.height)
        if generate_intersections:
            self.variants = add_join_variants(self.variants)
        print(colored("Number of variants: {}".format(len(self.variants)), 'green'))

    def get_graphs(self):
        graph_list = []
        for variant in self.variants:
            sequence = convert_solution_to_join_sequence(variant)
            graph = sequence.generate_graph()
            graph_list.append(graph)
        return graph_list

    def get_animations(self, scale=1, ratio=None, spacing=None):
        if spacing is None:
            spacing = [1, 1]
        if ratio is None:
            ratio = [16, 9]
        helper = GridShowCase(num_elements=len(self.variants),
                              element_dimensions=(scale * self.width, scale * self.height),
                              spacing=spacing, space_ratio=ratio)
        animations_list = []
        graph_list = []
        for index, variant in enumerate(self.variants):
            shift = helper.get_element_coords(index)
            sequence = convert_solution_to_join_sequence(variant)
            animations, graph = sequence.get_animations(scale=scale, shift=shift)
            animations_list.append(animations)
            graph_list.append(graph)

        return animations_list, graph_list, helper


class JoinSequence:
    def __init__(self, width, height, sequence):
        self.width = width
        self.height = height
        self.sequence = sequence

    def __str__(self):
        _str = "Join Sequence: "
        for idx, (x, y, operation) in enumerate(self.sequence):
            _str += "{}.({}|{}, {});".format(idx, x, y, operation)
        return _str

    def generate_graph(self, scale=1, shift=[0, 0]):
        graph = Graph(width=self.width, height=self.height, scale=scale, shift=shift)
        graph.init_cycles()
        for (x, y, operation) in self.sequence:
            searcher = GraphSearcher(graph)
            joint = searcher.evaluate_position((x, y))
            if operation == 'intersect':
                joint.intersect()
            elif operation == 'merge':
                joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))
        return graph

    def get_animations(self, scale, shift):
        animations = []
        graph = Graph(width=self.width, height=self.height, scale=scale, shift=shift)
        animations += draw_graph(graph)
        animations += make_unitary(graph)
        graph.init_cycles()
        for (x, y, operation) in self.sequence:
            searcher = GraphSearcher(graph)
            # joints = searcher.walk_graph()
            # animations.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))
            # joint = joints[index]
            joint = searcher.evaluate_position((x, y))
            # animations.append(AnimationObject(type='remove', content=joint.drawable))
            if operation == 'intersect':
                animations += joint.intersect()
            elif operation == 'merge':
                animations += joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))
            # animations.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints], wait_after=1))
        return animations, graph


def convert_solution_to_join_sequence(ip_solution):
    width = len(ip_solution)
    height = len(ip_solution[0])
    sequence = []
    for x in range(0, width, 2):
        for y in range(1, height, 2):
            if ip_solution[x][y] == 1:
                sequence.append((x, y, 'merge'))
            elif ip_solution[x][y] == 2:
                sequence.append((x, y, 'intersect'))
    for y in range(0, height, 2):
        for x in range(1, width, 2):
            if ip_solution[x][y] == 1:
                sequence.append((x, y, 'merge'))
            elif ip_solution[x][y] == 2:
                sequence.append((x, y, 'intersect'))
    for x in range(1, width, 2):
        for y in range(1, height, 2):
            if ip_solution[x][y] == 1:
                sequence.append((x, y, 'merge'))
            elif ip_solution[x][y] == 2:
                sequence.append((x, y, 'intersect'))
    return JoinSequence(width + 1, height + 1, sequence)


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


def generate_join_variants(solution):
    variants = []
    intersect_matrix, n = get_intersect_matrix(solution)
    combinations = [list(i) for i in itertools.product([0, 1], repeat=n)]
    non_zero_indices = np.argwhere(intersect_matrix > 0)
    for variation in combinations:
        variant = np.copy(intersect_matrix)
        for index, joint_decision in enumerate(variation):
            x, y = non_zero_indices[index]
            variant[x][y] = joint_decision

        variant = variant + np.array(solution)
        variants.append(variant)

    return variants


def add_join_variants(solutions):
    complete_list = []
    for solution in solutions:
        complete_list += generate_join_variants(solution)
    return complete_list


def get_problem(graph_width, graph_height):
    return Problem(graph_width - 1, graph_height - 1)


def iterate(width, height, _print=False):
    cells = []
    raster = []
    free = []
    for x in range(width):
        for y in range(height):
            cells.append([x, y])
            if x % 2 == 0 and y % 2 == 0:
                raster.append([x, y])
            else:
                free.append([x, y])

    n_cells = width * height
    n_raster = int(np.ceil(width / 2) * np.ceil(height / 2))
    n_free = n_cells - n_raster
    n_choice = n_raster - 1
    # n_variants = np.prod(np.arange(n_free, n_free - n_choice + 1, -1))  # wrong

    assert n_cells == len(cells)
    assert n_raster == len(raster)
    assert n_free == len(free)

    if _print:
        print("Total Cells: {}, Occupied by Raster: {}, Free : {}, Choice: {}".format(n_cells, n_raster, n_free, n_choice))

    queue = [[i] for i in range(n_free - n_choice + 1)]
    variants = []
    counter = 0

    while len(queue) > 0:
        if _print:
            print("Iteration {}, queue size: {}".format(counter, len(queue)))
        sequence = queue.pop(0)
        _free = np.arange(sequence[-1] + 1, n_free, 1)  # since order is irrelevant, we always choose elements in ascending order
        # print("Parent sequence: {}, available options: {}".format(sequence, _free))
        for cell_index in _free:
            _sequence = sequence + [cell_index]
            if n_choice - len(_sequence) > n_free - cell_index:  # not enough free spaces left to choose a full sequence
                continue
            # print("Child sequence: {}".format(_sequence))
            # problem = Problem(width, height, [free[i] for i in _sequence])
            problem = Problem(width, height, [free[i] for i in _sequence])
            solution, status = problem.solve()
            # solution, status = (0, 1)
            if not status:
                continue
            elif len(_sequence) >= n_choice:
                variants.append(solution)
            else:
                queue.append(_sequence)

        counter += 1

    return variants


if __name__ == '__main__':
    GraphModel(6, 6)
