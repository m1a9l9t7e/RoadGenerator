import random
import sys
import time
from tqdm import tqdm
from graph import Graph, GraphSearcher
from ip.ip_util import get_intersect_matrix, QuantityConstraint, TrackProperties, ConditionTypes, get_grid_indices, list_grid_as_str
from ip.problem import Problem, IntersectionProblem
from util import GridShowCase, get_adjacent, print_2d
import numpy as np
import itertools
from termcolor import colored
import multiprocessing as mp


class GraphModel:
    def __init__(self, width, height, generate_intersections=True, intersections_ip=True, sample_random=None):
        self.width = width - 1
        self.height = height - 1
        iterator = GGMSTIterator(self.width, self.height, _print=False)
        start = time.time()
        self.variants = iterator.iterate()
        end = time.time()

        if generate_intersections:
            print(colored("Number of variants w/o intersections: {}".format(len(self.variants)), 'green'))
            print(colored("Time elapsed: {:.2f}s".format(end - start), 'blue'))
            start = time.time()
            if intersections_ip:
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


class GGMSTIterator:
    def __init__(self, width, height, _print=False):
        self.width = width
        self.height = height
        self._print = _print
        self.variants = []
        self.counter = 0

        cells = []
        corners = []
        self.free = []
        n_cells = width * height

        for x in range(width):
            for y in range(height):
                cells.append([x, y])
                if x % (width-1) == 0 and y % (height-1) == 0:  # only corners
                    corners.append([x, y])
                else:
                    self.free.append([x, y])

        n_preoccupied = len(corners)
        self.n_free = n_cells - n_preoccupied
        self.n_choice = int(np.ceil(width / 2) * np.ceil(height / 2)) * 2 - 1 - n_preoccupied

        if _print:
            print("Total Cells: {}, Occupied by Corners: {}, Free : {}, Choice: {}".format(n_cells, n_preoccupied, self.n_free, self.n_choice))

    def iterate(self, multi_processing=True):
        queue = [[i] for i in range(self.n_free - self.n_choice + 1)]

        with tqdm(total=len(queue)) as pbar:
            while len(queue) > 0:
                # if self._print:
                    # print("Processed elements {}, queue size: {}".format(self.counter, len(queue)))
                if multi_processing:
                    pool = mp.Pool(mp.cpu_count())
                    full_iteration = pool.map(self.next, queue)
                    pool.close()
                    queue = []
                    for next_elements in full_iteration:
                        add_to_queue = self.unpack_next(next_elements)
                        queue += add_to_queue
                        self.counter += len(add_to_queue)
                        pbar.update(len(add_to_queue))

                else:
                    next_elements = self.next(queue.pop(0))
                    add_to_queue = self.unpack_next(next_elements)
                    queue += add_to_queue
                    self.counter += 1
                    pbar.update(1)
                pbar.total = len(queue)
                pbar.refresh()

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
            problem = Problem(self.width, self.height, iteration_constraints=[self.free[i] for i in _sequence])
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


# TODO read straight length from dict instead of from method call?
def convert_solution_to_graph(ip_solution, problem_dict={}, straight_length=3, shift=[0, 0], scale=1, get_intersections=False):
    width = len(ip_solution)
    height = len(ip_solution[0])
    edge_list = []

    # construct base graph
    for x in range(width):
        for y in range(height):
            if ip_solution[x][y] > 0:
                adjacent_cells = get_adjacent(ip_solution, (x, y))
                adjacent_edges = get_edges_adjacent_to_cell((x, y))
                for index, adjacent in enumerate(adjacent_cells):
                    if not adjacent_cells[index]:
                        edge_list.append(adjacent_edges[index])
    graph = Graph(width+1, height+1, edge_list=edge_list, shift=shift, scale=scale)

    # construct and mark intersections
    intersections = []
    for x in range(width):
        for y in range(height):
            if ip_solution[x][y] == 2:
                searcher = GraphSearcher(graph)
                intersection = searcher.evaluate_position((x, y), ignore_cycles=True)
                intersections.append(intersection)
                for (_x, _y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    graph.grid[x + _x][y + _y].track_property = TrackProperties.intersection

    # Mark all nodes with track properties
    nodes = graph.grid

    # mark straights
    if 'horizontal_straights' in problem_dict.keys():
        horizontal_straights = problem_dict['horizontal_straights']
        vertical_straights = problem_dict['vertical_straights']
        for (x, y) in get_grid_indices(width, height):
            bottom, top = horizontal_straights[x][y]
            if bottom > 0:
                for i in range(straight_length):
                    nodes[x + i][y].track_property = TrackProperties.straight
            if top > 0:
                for i in range(straight_length):
                    nodes[x + i][y + 1].track_property = TrackProperties.straight
            left, right = vertical_straights[x][y]
            if left > 0:
                for i in range(straight_length):
                    nodes[x][y + i].track_property = TrackProperties.straight
            if right > 0:
                for i in range(straight_length):
                    nodes[x + 1][y + i].track_property = TrackProperties.straight

    # mark 90s
    if '90s_inner' in problem_dict.keys():
        _90s_inner = problem_dict['90s_inner']
        _90s_outer = problem_dict['90s_outer']
        for (x, y) in get_grid_indices(width, height):
            bottom_right, top_right, top_left, bottom_left = _90s_inner[x][y]
            if bottom_left > 0:
                nodes[x][y].track_property = TrackProperties.turn_90
            if bottom_right > 0:
                nodes[x+1][y].track_property = TrackProperties.turn_90
            if top_left > 0:
                nodes[x][y+1].track_property = TrackProperties.turn_90
            if top_right > 0:
                nodes[x+1][y+1].track_property = TrackProperties.turn_90
            bottom_right, top_right, top_left, bottom_left = _90s_outer[x][y]
            if bottom_left > 0:
                nodes[x][y].track_property = TrackProperties.turn_90
            if bottom_right > 0:
                nodes[x+1][y].track_property = TrackProperties.turn_90
            if top_left > 0:
                nodes[x][y+1].track_property = TrackProperties.turn_90
            if top_right > 0:
                nodes[x+1][y+1].track_property = TrackProperties.turn_90

    # mark 180s
    if '180s_inner' in problem_dict.keys():
        _180s_inner = problem_dict['180s_inner']
        _180s_outer = problem_dict['180s_outer']
        for (x, y) in get_grid_indices(width, height):
            right, top, left, bottom = _180s_outer[x][y]
            if bottom > 0:
                nodes[x][y].track_property = TrackProperties.turn_180
                nodes[x+1][y].track_property = TrackProperties.turn_180
            if left > 0:
                nodes[x][y].track_property = TrackProperties.turn_180
                nodes[x][y+1].track_property = TrackProperties.turn_180
            if top > 0:
                nodes[x][y+1].track_property = TrackProperties.turn_180
                nodes[x+1][y+1].track_property = TrackProperties.turn_180
            if right > 0:
                nodes[x+1][y].track_property = TrackProperties.turn_180
                nodes[x+1][y+1].track_property = TrackProperties.turn_180
            right_top, right_bottom, top_right, top_left = _180s_inner[x][y]
            if right_bottom > 0:
                nodes[x+1][y].track_property = TrackProperties.turn_180
                nodes[x+2][y].track_property = TrackProperties.turn_180
            if right_top > 0:
                nodes[x+1][y+1].track_property = TrackProperties.turn_180
                nodes[x+2][y+1].track_property = TrackProperties.turn_180
            if top_left > 0:
                nodes[x][y+1].track_property = TrackProperties.turn_180
                nodes[x][y+2].track_property = TrackProperties.turn_180
            if top_right > 0:
                nodes[x+1][y+1].track_property = TrackProperties.turn_180
                nodes[x+1][y+2].track_property = TrackProperties.turn_180

    if get_intersections:
        return graph, intersections
    else:
        for intersection in intersections:
            intersection.intersect()
        return graph


def get_edges_adjacent_to_cell(coords):
    x, y = coords
    edge_bottom = ((x, y), (x+1, y))
    edge_left = ((x, y), (x, y+1))
    edge_right = ((x+1, y), (x+1, y+1))
    edge_top = ((x, y+1), (x+1, y+1))

    return [edge_right, edge_top, edge_left, edge_bottom]


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


def get_custom_solution(width, height, quantity_constraints=[], iteration_constraints=None, print_stats=True):
    # Get Solution
    start = time.time()
    problem = Problem(width - 1, height - 1, iteration_constraints=iteration_constraints, quantity_constraints=quantity_constraints)
    solution, status = problem.solve(_print=False, print_zeros=False)
    end = time.time()
    print(colored("Solution {}, Time elapsed: {:.2f}s".format('optimal' if status > 1 else 'infeasible', end - start), "green" if status > 1 else "red"))
    if status <= 0:
        sys.exit(0)

    if print_stats:
        problem.get_stats()
    return solution, problem.export_variables()


def get_problem(graph_width, graph_height):
    return Problem(graph_width - 1, graph_height - 1)


if __name__ == '__main__':
    # 6x6 complete with intersections: 52960
    # 8x4 complete with intersections: 12796
    GraphModel(6, 6, generate_intersections=False)
    # solution = get_custom_solution(4, 4)
