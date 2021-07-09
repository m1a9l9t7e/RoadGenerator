import random
import time

from graph import Graph, GraphSearcher
from ip_reformulation import GGMSTProblem, IntersectionProblem
from util import draw_graph, make_unitary, GridShowCase, get_adjacent
import numpy as np
import itertools
from termcolor import colored
import multiprocessing as mp


class GraphModel:
    def __init__(self, width, height, generate_intersections=True, fast=False, ip_intersections=True, sample_random=None):
        self.width = width - 1
        self.height = height - 1
        iterator = GGMSTIterator(self.width, self.height, raster=fast, _print=True)
        start = time.time()
        self.variants = iterator.iterate()
        end = time.time()

        if generate_intersections:
            print(colored("Number of variants w/o intersections: {}".format(len(self.variants)), 'green'))
            print(colored("Time elapsed: {:.2f}s".format(end - start), 'blue'))
            start = time.time()
            if ip_intersections:
                self.variants += get_join_variants_ip(self.variants)
            else:
                self.variants = add_join_variants(self.variants)
            end = time.time()

        print(colored("Number of variants: {}".format(len(self.variants)), 'green'))
        print(colored("Time elapsed: {:.2f}s".format(end - start), 'blue'))

        if sample_random is not None and sample_random < len(self.variants):
            self.variants = random.sample(self.variants, sample_random)
            print(colored("Random samples: {}".format(len(self.variants)), 'green'))

    def get_graphs(self, scale=1, ratio=[16, 9], spacing=[1, 1]):
        helper = GridShowCase(num_elements=len(self.variants),
                              element_dimensions=(scale * self.width, scale * self.height),
                              spacing=spacing, space_ratio=ratio)
        graph_list = []
        for index, variant in enumerate(self.variants):
            shift = helper.get_element_coords(index)
            graph = convert_solution_to_graph(variant, scale=scale, shift=shift)
            graph_list.append(graph)
        return graph_list, helper

    def get_animations(self, scale=1, ratio=[16, 9], spacing=[1, 1]):
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


class GGMSTIterator:
    def __init__(self, width, height, raster=True, _print=False):
        self.width = width
        self.height = height
        self.raster = raster
        self._print = _print
        self.variants = []
        self.counter = 0

        cells = []
        raster = []
        self.free = []
        n_cells = width * height

        if raster:
            for x in range(width):
                for y in range(height):
                    cells.append([x, y])
                    if x % 2 == 0 and y % 2 == 0:  # complete raster
                        raster.append([x, y])
                    else:
                        self.free.append([x, y])

            n_preoccupied = len(raster)
            self.n_free = n_cells - n_preoccupied
            self.n_choice = n_preoccupied - 1
        else:
            for x in range(width):
                for y in range(height):
                    cells.append([x, y])
                    if x % (width-1) == 0 and y % (height-1) == 0:  # only corners
                        raster.append([x, y])
                    else:
                        self.free.append([x, y])

            n_preoccupied = len(raster)
            self.n_free = n_cells - n_preoccupied
            self.n_choice = int(np.ceil(width / 2) * np.ceil(height / 2)) * 2 - 1 - n_preoccupied

        if _print:
            print("Total Cells: {}, Occupied by Raster: {}, Free : {}, Choice: {}".format(n_cells, n_preoccupied, self.n_free, self.n_choice))

    def iterate(self, multi_processing=True):
        queue = [[i] for i in range(self.n_free - self.n_choice + 1)]

        while len(queue) > 0:
            if self._print:
                print("Processed elements {}, queue size: {}".format(self.counter, len(queue)))
            if multi_processing:
                pool = mp.Pool(mp.cpu_count())
                full_iteration = pool.map(self.next, queue)
                pool.close()
                queue = []
                for next_elements in full_iteration:
                    add_to_queue = self.unpack_next(next_elements)
                    queue += add_to_queue
                    self.counter += len(add_to_queue)

            else:
                next_elements = self.next(queue.pop(0))
                add_to_queue = self.unpack_next(next_elements)
                queue += add_to_queue
                self.counter += 1

        return self.variants

    def next(self, sequence):
        _next = []
        _free = np.arange(sequence[-1] + 1, self.n_free, 1)  # since order is irrelevant, we always choose elements in ascending order
        # print("Parent sequence: {}, available options: {}".format(sequence, _free))
        for cell_index in _free:
            _sequence = sequence + [cell_index]
            if self.n_choice - len(_sequence) > self.n_free - cell_index:  # not enough free spaces left to choose a full sequence
                continue
            # print("Child sequence: {}".format(_sequence))
            # problem = Problem(width, height, [free[i] for i in _sequence])
            problem = GGMSTProblem(self.width, self.height, [self.free[i] for i in _sequence])
            solution, feasible = problem.solve()
            # solution, status = (0, 1)
            if feasible:
                if len(_sequence) >= self.n_choice:
                    _next.append((True, solution))
                else:
                    _next.append((False, _sequence))

        return _next

    def unpack_next(self, next_elements):
        add_to_queue = []
        for element in next_elements:
            leaf, content = element
            if leaf:
                self.variants.append(content)
            else:
                add_to_queue.append(content)

        return add_to_queue


class IntersectionIterator:
    def __init__(self, non_zero_indices, allow_adjacent=False, n=None, _print=False):
        self.non_zero_indices = non_zero_indices
        self.n = n
        self.allow_adjacent = allow_adjacent
        self._print = _print
        self.counter = 0
        self.solutions = []

    def iterate(self, multi_processing=True):
        queue = [[]]

        while len(queue) > 0:
            if self._print:
                print("Processed elements {}, queue size: {}".format(self.counter, len(queue)))
            if multi_processing:
                pool = mp.Pool(mp.cpu_count())
                full_iteration = pool.map(self.next, queue)
                pool.close()
                queue = []
                for next_elements in full_iteration:
                    add_to_queue = self.unpack_next(next_elements)
                    queue += add_to_queue
                    self.counter += len(add_to_queue)

            else:
                next_elements = self.next(queue.pop(0))
                add_to_queue = self.unpack_next(next_elements)
                queue += add_to_queue
                self.counter += 1

        return self.solutions

    def next(self, sequence):
        _next = []
        if len(sequence) == 0:
            last_element = -1
        else:
            last_element = sequence[-1]
        next_choices = np.arange(last_element + 1, len(self.non_zero_indices), 1)  # since order is irrelevant, we always choose elements in ascending order
        # print("Parent sequence: {}, available options: {}".format(sequence, next_choices))
        for index in next_choices:
            _sequence = sequence + [index]
            # print("Evaluating sequence: {}".format(_sequence))
            if self.n is not None:
                if len(self.non_zero_indices) - index < self.n - len(_sequence):  # not enough indices left to satisfy n
                    # print("PREEMPTIVE DENY")
                    continue
                if len(_sequence) > self.n:  # sequence has more intersection than n
                    # print("PREEMPTIVE DENY")
                    continue
            problem = IntersectionProblem(self.non_zero_indices, n=self.n, allow_adjacent=self.allow_adjacent, extra_constraints=_sequence)
            solution, feasible = problem.solve()
            # solution, feasible = ([0], 1)
            if feasible:
                # print("FEASIBLE")
                if self.n is None:
                    _next.append((True, solution))
                    _next.append((False, _sequence))
                else:
                    if len(_sequence) == self.n:
                        _next.append((True, solution))
                    elif len(_sequence) < self.n:
                        _next.append((False, _sequence))
            # else:
                # print("NOT FEASIBLE")

        return _next

    def unpack_next(self, next_elements):
        add_to_queue = []
        for element in next_elements:
            leaf, content = element
            if leaf:
                self.solutions.append(content)
            else:
                add_to_queue.append(content)

        return add_to_queue


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


def convert_solution_to_graph(ip_solution, shift=[0, 0], scale=1):
    width = len(ip_solution)
    height = len(ip_solution[0])
    edge_list = []

    for x in range(width):
        for y in range(height):
            if ip_solution[x][y] > 0:
                adjacent_cells = get_adjacent(ip_solution, (x, y))
                adjacent_edges = get_edges_adjacent_to_cell((x, y))
                for index, adjacent in enumerate(adjacent_cells):
                    if not adjacent_cells[index]:
                        edge_list.append(adjacent_edges[index])
    graph = Graph(width+1, height+1, edge_list=edge_list, shift=shift, scale=scale)
    for x in range(width):
        for y in range(height):
            if ip_solution[x][y] == 2:
                searcher = GraphSearcher(graph)
                joint = searcher.evaluate_position((x, y), ignore_cycles=True)
                joint.intersect()
    return graph


def get_edges_adjacent_to_cell(coords):
    x, y = coords
    edge_bottom = ((x, y), (x+1, y))
    edge_left = ((x, y), (x, y+1))
    edge_right = ((x+1, y), (x+1, y+1))
    edge_top = ((x, y+1), (x+1, y+1))

    return [edge_right, edge_top, edge_left, edge_bottom]


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


def generate_join_variants(solution, n_intersections=None, allow_adjacent_intersections=False, allow_intersect_at_stubs=False):
    variants = []
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs)
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


def generate_join_variants_ip(solution, n_intersections=None, allow_adjacent_intersections=False, allow_intersect_at_stubs=False):
    variants = []
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs)
    non_zero_indices = np.argwhere(intersect_matrix > 0)
    iterator = IntersectionIterator(non_zero_indices, allow_adjacent=allow_adjacent_intersections, n=n_intersections, _print=False)
    combinations = iterator.iterate()

    for variation in combinations:
        variant = np.copy(intersect_matrix)
        for index, joint_decision in enumerate(variation):
            x, y = non_zero_indices[index]
            variant[x][y] = joint_decision

        variant = variant + np.array(solution)
        variants.append(variant)

    return variants


def add_join_variants(solutions, allow_intersect_at_stubs=False):
    complete_list = []
    for solution in solutions:
        complete_list += generate_join_variants(solution, allow_intersect_at_stubs=allow_intersect_at_stubs)
    return complete_list


def get_join_variants_ip(solutions, allow_intersect_at_stubs=False, n_intersections=None):
    join_variants = []
    for solution in solutions:
        join_variants += generate_join_variants_ip(solution, n_intersections=n_intersections, allow_intersect_at_stubs=allow_intersect_at_stubs)
    return join_variants


def get_problem(graph_width, graph_height):
    return GGMSTProblem(graph_width - 1, graph_height - 1)


#######################
####### IP UTIL ######
#######################

def get_random_solution(width, height):
    # Get Solution
    problem = get_problem(width, height)
    solution, status = problem.solve(_print=False)

    # Add intersections
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs=False)
    non_zero_indices = np.argwhere(intersect_matrix > 0)
    for index in range(n):
        x, y = non_zero_indices[index]
        intersect_matrix[x][y] = random.choice([0, 1])
    solution = intersect_matrix + np.array(solution)
    return solution


def get_custom_solution(width, height, wishes=None):
    if wishes is None:
        wishes = {
            'n_intersections': 2,
            'allow_adjacent_intersections': False,
            'allow_intersect_at_stubs': False,
            # 'n_straights': 2,
            # 'n_90_degree_turns': 3,
            # 'n_180_degree_turns': 1,
            'hard_constraints': [[0, 1], [1, 2], [2, 1]]
        }
    # Get Solution
    start = time.time()
    ggmst_problem = GGMSTProblem(width - 1, height - 1, extra_constraints=wishes['hard_constraints'])
    solution, status = ggmst_problem.solve(_print=False, print_zeros=False)
    end = time.time()
    print(colored('GGMST solved in {:.2f}s. {}easible solution found!'.format(end - start, 'No f' if status < 1 else 'F'), 'green'))

    # Add intersections
    start = time.time()
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs=wishes['allow_intersect_at_stubs'])
    non_zero_indices = np.argwhere(intersect_matrix > 0)
    n_intersections = random.randint(0, n) if wishes['n_intersections'] is None else wishes['n_intersections']
    intersection_problem = IntersectionProblem(non_zero_indices, n=n_intersections, allow_adjacent=wishes['allow_adjacent_intersections'])
    selection, status = intersection_problem.solve()
    end = time.time()
    print(colored('Intersections solved in {:.2f}s. {}easible solution found!'.format(end - start, 'No f' if status < 1 else 'F'), 'green'))

    for index in range(n):
        x, y = non_zero_indices[index]
        intersect_matrix[x][y] = selection[index]
    solution = intersect_matrix + np.array(solution)

    return solution


if __name__ == '__main__':
    # 6x6 complete with intersections: 52960
    # 8x4 complete with intersections: 12796
    GraphModel(4, 4, generate_intersections=False, fast=False)
    # solution = get_custom_solution(4, 4)
