import random
import sys
import time
from tqdm import tqdm
from graph import Graph, GraphSearcher
from ip.ip_util import get_intersect_matrix, QuantityConstraint, ConditionTypes, get_grid_indices, list_grid_as_str, QuantityConstraintStraight, parse_ip_config, SolutionEntries
from ip.problem import Problem, IntersectionProblem
from ip.iterative_construction import Iterator as IterativeConstructionIterator
from ip.zones import ZoneProblem
from util import GridShowCase, get_adjacent, get_adjacent_bool, TrackProperties, max_adjacent, extract_graph_tours, ZoneTypes, print_2d, ZoneMisc
import numpy as np
import itertools
from termcolor import colored
import multiprocessing as mp
from enum import Enum, auto


class IteratorType(Enum):
    base_ip = auto()
    prohibition_ip = auto()
    iterative_construction = auto()


class GraphModel:
    def __init__(self, width, height, generate_intersections=True, intersections_ip=False, sample_random=None, allow_gap_intersections=True,
                 allow_adjacent_intersections=False, iterator_type=IteratorType.iterative_construction, just_count=False):
        self.width = width - 1
        self.height = height - 1

        start = time.time()

        if iterator_type == IteratorType.base_ip:
            iterator = GGMSTIterator(self.width, self.height, _print=False)
            self.variants = iterator.iterate()
        elif iterator_type == IteratorType.prohibition_ip:
            iterator = ProhibitionIterator(self.width, self.height, _print=True)
            self.variants = iterator.iterate()
        elif iterator_type == IteratorType.iterative_construction:
            iterator = IterativeConstructionIterator(self.width, self.height, _print=False)
            if self.width == 7 and self.height == 7:
                self.variants = np.load('/home/malte/PycharmProjects/circuit-creator/ip/7x7_complete/variants.npy')
            else:
                self.variants = iterator.iterate(multi_processing=False, continued=False)
        else:
            raise ValueError("Unkown Iterator Type: {}".format(iterator_type))

        end = time.time()

        if generate_intersections:
            print(colored("Number of variants w/o intersections: {}".format(len(self.variants)), 'green'))
            print(colored("Time elapsed: {:.2f}s".format(end - start), 'blue'))
            start = time.time()
            if just_count:
                num_variants = _count_intersection_variants(self.variants)
                print(colored("Estimated number of variants: {}".format(num_variants), 'green'))
                return
            elif intersections_ip:
                self.variants += add_intersection_variants_ip(self.variants, allow_gap_intersection=allow_gap_intersections,
                                                              allow_adjacent_intersections=allow_adjacent_intersections)
            else:
                self.variants = get_intersection_variants(self.variants, allow_gaps=allow_gap_intersections, allow_adjacent=allow_adjacent_intersections)
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

    def get_meta_info(self):
        meta_info = []
        for variant in self.variants:
            graph = convert_solution_to_graph(variant, scale=1, shift=[0, 0])
            info = VariantMetaInfo(number_gap_intersections=np.count_nonzero(variant == 3), drive_straight=extract_graph_tours(graph) == 1)
            meta_info.append(info)
        return meta_info


class VariantMetaInfo:
    def __init__(self, number_gap_intersections, drive_straight):
        self.number_gap_intersections = number_gap_intersections
        self.drive_straight = drive_straight


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

        with tqdm(total=0) as pbar:
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


class ProhibitionIterator:
    def __init__(self, width, height, _print=False):
        self.width = width
        self.height = height
        self._print = _print
        self.variants = []
        self.counter = 0

    def iterate(self):
        solutions = []
        while True:
            problem = Problem(self.width, self.height, prohibition_constraints=solutions)
            start = time.time()
            solution, feasible = problem.solve()
            end = time.time()
            if self._print:
                print(colored("Found Solution {}. Time elapsed: {:.2f}s".format(len(solutions), end - start), 'cyan'))
            if feasible:
                solutions.append(self.flatten(solution))
                self.variants.append(solution)
            else:
                return self.variants

    def flatten(self, solution):
        flat = []
        for x in range(self.width):
            for y in range(self.height):
                if solution[x][y]:
                    flat.append((x, y))

        return flat


class IntersectionIterator:
    def __init__(self, intersection_indices, gap_intersection_indices=None, allow_adjacent=False, n=None, _print=False):
        self.intersection_indices = intersection_indices
        if gap_intersection_indices is None:
            self.gap_intersection_indices = []
        else:
            self.gap_intersection_indices = gap_intersection_indices

        self.allow_adjacent = allow_adjacent
        self.n = n
        self._print = _print
        self.counter = 0
        self.solutions = []

    def iterate(self, multi_processing=False):
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
        next_choices = np.arange(last_element + 1, len(self.intersection_indices) + len(self.gap_intersection_indices), 1)  # since order is irrelevant, we always choose elements in ascending order
        # print("Parent sequence: {}, available options: {}".format(sequence, next_choices))
        for index in next_choices:
            _sequence = sequence + [index]
            # print("Evaluating sequence: {}".format(_sequence))
            if self.n is not None:
                if len(self.intersection_indices) - index < self.n - len(_sequence):  # not enough indices left to satisfy n
                    # print("PREEMPTIVE DENY")
                    continue
                if len(_sequence) > self.n:  # sequence has more intersection than n
                    # print("PREEMPTIVE DENY")
                    continue
            problem = IntersectionProblem(self.intersection_indices, gap_intersection_indices=self.gap_intersection_indices,
                                          n=self.n, allow_adjacent=self.allow_adjacent, iteration_constraints=_sequence)
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


def convert_solution_to_graph(ip_solution, problem_dict={}, shift=[0, 0], scale=1, get_intersections=False):
    width = len(ip_solution)
    height = len(ip_solution[0])
    edge_list = []

    # construct base graph
    for x in range(width):
        for y in range(height):
            if ip_solution[x][y] > 0:
                adjacent_cells = get_adjacent_bool(ip_solution, (x, y))
                adjacent_edges = get_edges_adjacent_to_cell((x, y))
                for index, adjacent in enumerate(adjacent_cells):
                    if not adjacent_cells[index]:
                        edge_list.append(adjacent_edges[index])
    graph = Graph(width+1, height+1, edge_list=edge_list, shift=shift, scale=scale)

    # construct and mark intersections
    intersections = []
    for x in range(width):
        for y in range(height):
            if ip_solution[x][y] in [SolutionEntries.negative_and_intersection, SolutionEntries.positive_and_intersection]:
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
            for straight_length in range(2, width):
                if straight_length not in horizontal_straights[x][y].keys():
                    continue

                bottom, top = horizontal_straights[x][y][straight_length]
                if bottom > 0:
                    for i in range(straight_length):
                        nodes[x + i][y].track_property = TrackProperties.straight
                if top > 0:
                    for i in range(straight_length):
                        nodes[x + i][y + 1].track_property = TrackProperties.straight
                left, right = vertical_straights[x][y][straight_length]
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


def count_intersection_variants(solution):
    intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_gap=False, allow_intersect_at_stubs=False)
    return 2**n


def generate_intersection_variants_ip(solution, n_intersections=None, allow_adjacent_intersections=False, allow_gap_intersections=False):
    variants = []
    if allow_gap_intersections:
        all_intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_gap=True)
        intersect_matrix = np.where(np.array(solution) > 0, all_intersect_matrix, np.zeros_like(solution))
        gap_intersect_matrix = np.where(np.array(solution) == 0, all_intersect_matrix, np.zeros_like(solution))
        gap_intersection_indices = np.argwhere(gap_intersect_matrix > 0)
    else:
        intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_gap=False)
        gap_intersection_indices = []

    intersection_indices = np.argwhere(intersect_matrix > 0)
    iterator = IntersectionIterator(intersection_indices, gap_intersection_indices=gap_intersection_indices,
                                    allow_adjacent=allow_adjacent_intersections, n=n_intersections, _print=False)

    combinations = iterator.iterate()

    for variation in combinations:
        variant = np.copy(intersect_matrix)
        for index, joint_decision in enumerate(variation):
            x, y = intersection_indices[index]
            variant[x][y] = joint_decision

        variant = variant + np.array(solution)
        variants.append(variant)

    return variants


def _count_intersection_variants(solutions):
    pool = mp.Pool(mp.cpu_count())
    complete_list = pool.map(count_intersection_variants, tqdm(solutions, desc='Counting Intersections'))
    pool.close()
    num_intersect_variants = sum(complete_list)
    return num_intersect_variants


def get_intersection_variants(solutions, allow_adjacent, allow_gaps):
    duplicate_lookup = dict()
    duplicate_counter = 0
    intersection_variants = []
    for solution in tqdm(solutions, desc='Generating Intersections'):
        all_intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_gap=True)

        # get intersection positions
        intersect_matrix = np.where(np.array(solution) > 0, all_intersect_matrix, np.zeros_like(solution))
        intersection_indices = np.argwhere(intersect_matrix > 0)

        # get gap intersection positions
        gap_intersect_matrix = np.where(np.array(solution) == 0, all_intersect_matrix, np.zeros_like(solution))
        if allow_gaps:
            gap_intersection_indices = np.argwhere(gap_intersect_matrix > 0)
        else:
            gap_intersection_indices = []

        # get variants without gap intersections
        _intersection_variants, _duplicate_counter = generate_intersection_variants(solution, intersection_indices, gap_intersection_indices, duplicate_lookup, allow_adjacent=allow_adjacent)
        intersection_variants += _intersection_variants
        duplicate_counter += _duplicate_counter
        # get variants with gap intersections
        # this requires more effort to avoid duplicates
        # specifically, possibilities are iterated starting from a single gap intersection, which is checked for duplicates
        # for index, gap_intersection in enumerate(gap_intersection_indices):
        #     variant = np.copy(solution)
        #     variant[gap_intersection[0]][gap_intersection[1]] = SolutionEntries.negative_and_intersection
        #     hashed = hash_np(variant)
        #     if hashed in duplicate_lookup:
        #         duplicate_counter += 1
        #         continue
        #     else:
        #         duplicate_lookup[hashed] = True
        #         rm_gap_indices = list(gap_intersection_indices)
        #         del rm_gap_indices[index]
        #         intersection_variants += generate_intersection_variants_v2(variant, intersection_indices, rm_gap_indices, duplicate_lookup, allow_adjacent=allow_adjacent)

    # for intersection_variant in intersection_variants:
    #     print_2d(intersection_variant)
    # sys.exit(0)
    print(colored("{} duplicates found!".format(duplicate_counter), "yellow"))
    return intersection_variants


def generate_intersection_variants(solution, indices, gap_indices, duplicate_lookup, allow_adjacent=False):
    duplicate_counter = 0
    variants = []
    if len(gap_indices) > 0:
        all_indices = np.concatenate([indices, gap_indices])
    else:
        all_indices = indices

    # print("X indices:\n{}".format(indices))
    # print("X GAP indices:\n{}".format(gap_indices))
    # print("ALL X indices:\n{}".format(all_indices))
    combinations = [list(i) for i in itertools.product([0, 1], repeat=len(indices) + len(gap_indices))]
    skip = False
    for variation in combinations:
        variant = np.zeros_like(solution)
        enumeration = [(index, joint_decision) for index, joint_decision in enumerate(variation)]

        # Apply normal intersections first
        for index, joint_decision in enumeration[:len(indices)]:
            x, y = all_indices[index]
            if (not allow_adjacent) and max_adjacent(variant, (x, y)) > 0:
                skip = True
                break
            variant[x][y] = joint_decision

        if skip:
            skip = False
            continue

        # Apply gap intersections to allow adjacency checks
        for index, joint_decision in enumeration[len(indices):]:
            x, y = all_indices[index]
            if index >= len(indices):
                if allow_adjacent:
                    # check for adjacent non gap intersections
                    if max_adjacent(variant, (x, y)) == 1:
                        skip = True
                        break
                else:
                    # check for any adjacent intersections
                    if max(np.abs(get_adjacent(variant, (x, y)))) > 0:
                        skip = True
                        break
            else:
                print(colored("OH NOES", "red"))
                # This is a hack, to be able to use max(adjacent) == 1 to detect normal intersections exclusively and abs(adjacent) > 0 for all
                variant[x][y] = -joint_decision
        if skip:
            skip = False
            continue

        variant = np.where(variant == -1, SolutionEntries.negative_and_intersection, variant)
        variant = variant + np.array(solution)

        variation += [0]
        # contains gap intersections
        if np.max(variation[len(indices):]) > 0:
            # check for duplicate
            hashed = hash_np(variant)
            # if variant is a duplicate, we can return right away, because all further variants will have been reached
            # TODO: fix combination ordering, so this is actually true
            if hashed in duplicate_lookup:
                duplicate_counter += 1
                continue
            # else add duplicate to lookup dict
            else:
                duplicate_lookup[hash_np(variant)] = True

        variants.append(variant)

    return variants, duplicate_counter


def hash_np(arr):
    arr = np.where(arr == SolutionEntries.negative_and_intersection, SolutionEntries.positive_and_intersection, arr)
    arr.flags.writeable = False
    # return hash(arr.data)
    return hash(arr.tobytes())


def add_intersection_variants_ip(solutions, allow_adjacent_intersections=False, allow_gap_intersection=False, n_intersections=None):
    join_variants = []
    for solution in tqdm(solutions, desc='Generating Intersections'):
        join_variants += generate_intersection_variants_ip(solution, n_intersections=n_intersections,
                                                           allow_adjacent_intersections=allow_adjacent_intersections,
                                                           allow_gap_intersections=allow_gap_intersection)
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


def get_custom_solution(width, height, quantity_constraints=[], iteration_constraints=None, print_stats=True, allow_gap_intersections=True):
    # Get Solution
    start = time.time()
    problem = Problem(width - 1, height - 1, iteration_constraints=iteration_constraints, quantity_constraints=quantity_constraints,
                      allow_gap_intersections=allow_gap_intersections)
    solution, status = problem.solve(_print=False, print_zeros=False)
    end = time.time()
    print(colored("Solution {}, Time elapsed: {:.2f}s".format('optimal' if status > 1 else 'infeasible', end - start), "green" if status > 1 else "red"))
    if status <= 0:
        sys.exit(0)

    if print_stats:
        problem.get_stats()
    return solution, problem.export_variables()


def get_imitation_solution(original_solution, print_stats=False):
    quantity_constraints = [
        QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, 0),
        QuantityConstraint(TrackProperties.turn_180, ConditionTypes.more_or_equals, 0),
        QuantityConstraint(TrackProperties.turn_90, ConditionTypes.more_or_equals, 0)
    ]
    for length in range(2, len(original_solution)):
        quantity_constraints.append(QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=length, quantity=0))

    problem = Problem(len(original_solution), len(original_solution[0]), quantity_constraints=quantity_constraints, imitate=original_solution,
                      allow_adjacent_intersections=True, allow_gap_intersections=True)
    solution, status = problem.solve(_print=False, print_zeros=False)
    if status <= 0:
        raise ValueError("Original Solution Infeasible")
    if print_stats:
        problem.get_stats()
    return solution, problem.export_variables()


def get_problem(graph_width, graph_height):
    return Problem(graph_width - 1, graph_height - 1)


def get_solution_from_config(path, _print=True, allow_gap_intersections=True):
    dimensions, quantity_constraints = parse_ip_config(path)
    solution, _ = get_custom_solution(*dimensions, quantity_constraints=quantity_constraints, print_stats=_print, allow_gap_intersections=allow_gap_intersections)
    return solution


class ZoneDescription:
    def __init__(self, zone_type, min_length, max_length):
        self.zone_type = zone_type
        self.min_length = min_length
        self.max_length = max_length


def get_zone_solution(path_to_config, zone_descriptions, allow_gap_intersections=False):
    """
    Calcualte Zone Assignment in 3 Steps:
    1. Determine Start: The Start is set at the first parking zone. Parking zones are placed at matching straights.
       If there is no parking zone, the start is at node (0, 0). Describe Placement of Zones by indices. Index 0 is always at the start.
    2. Assign Expressway zones to remaining straights.
    3. Solve IP Problem for placement of urban and no passing.
    4. Shift indices so that description starts at (0, 0)! (This needs to be done so that zone descriptions match with indicees of graph tour)
    """
    dimensions, constraints = parse_ip_config(path_to_config)

    # change quantity constraints to accomodate parking and motorway
    new_constraints = constraints

    # solve problem with new constraints
    ip_solution, problem_dict = get_custom_solution(*dimensions, quantity_constraints=new_constraints, print_stats=True, allow_gap_intersections=allow_gap_intersections)
    graph = convert_solution_to_graph(ip_solution, problem_dict)
    graph_tour = extract_graph_tours(graph)[0]
    description_length = len(graph_tour.get_nodes())
    straight_zones = find_straight_zones(graph_tour)

    # decide which straights become parking/motorway
    straight_tuples = []
    for key in straight_zones.keys():
        straight_tuples += [(key, start, end) for (start, end) in straight_zones[key]]

    # shuffle straight tuples for more balanced selection?
    parking = []
    for zone_description in filter_zones(zone_descriptions, ZoneTypes.parking):
        for index, straight_tuple in enumerate(straight_tuples):
            (length, start, end) = straight_tuple
            if zone_description.min_length <= length <= zone_description.max_length:
                parking.append((start, end))
                straight_tuples.remove(straight_tuple)
                break

    # shuffle straight tuples for more balanced selection?
    express_way = []
    for zone_description in filter_zones(zone_descriptions, ZoneTypes.express_way):
        for index, straight_tuple in enumerate(straight_tuples):
            (length, start, end) = straight_tuple
            if zone_description.min_length <= length <= zone_description.max_length:
                express_way.append((start, end))
                straight_tuples.remove(straight_tuple)
                break

    blocked = parking + express_way
    # SET START TO FIRST PARKING ZONE AND SHIFT DESCRIPTION INDICEES
    if len(parking) == 0:
        start_index = 0
    else:
        start_index = parking[0][0]
        shifted = []
        for (start, end) in blocked:
            shifted.append(((start - start_index) % description_length, (end - start_index) % description_length))
        blocked = shifted

    # TODO -->
    # parking = [((start - start_index) % description_length, (end - start_index) % description_length) for (start, end) in parking]
    # TODO <--

    # Solve Zone Problem to place urban areas
    urban_area_descriptions = []
    for zone_description in filter_zones(zone_descriptions, ZoneTypes.urban_area):
        urban_area_descriptions.append((zone_description.min_length, zone_description.max_length))
    # Also add no-passing descriptions
    for zone_description in filter_zones(zone_descriptions, ZoneTypes.no_passing):
        urban_area_descriptions.append((zone_description.min_length, zone_description.max_length))

    p = ZoneProblem(zone_descriptions=urban_area_descriptions, blocked_zones=blocked, n=description_length)
    solution, status = p.solve()
    if status > 0:
        # solution = [(int(start), int(end)) for (start, end) in solution]
        solution = [(int(start), int(end)) for (start, end) in solution]
        solution = [((start + start_index) % description_length, (end + start_index) % description_length) for (start, end) in solution]

        p.show_solution()
        description_idx_to_zone_idx = p.get_solution_dict()

        urban_zones = []
        for description_idx, description in enumerate(filter_zones(zone_descriptions, ZoneTypes.urban_area)):
            urban_zone = solution[description_idx_to_zone_idx[description_idx]]
            urban_zones.append(urban_zone)
        no_passing_zones = []
        for description_idx, description in enumerate(filter_zones(zone_descriptions, ZoneTypes.no_passing)):
            description_idx += len(filter_zones(zone_descriptions, ZoneTypes.urban_area))
            no_passing_zone = solution[description_idx_to_zone_idx[description_idx]]
            no_passing_zones.append(no_passing_zone)
    else:
        print(colored("Selected Zones infeasible for Solution", 'red'))
        sys.exit(0)

    print("Selected Zones:\nParking: {}\nExpressways: {}\nUrban Areas: {}\nNo Passing: {}".format(colored(parking, 'cyan'), colored(express_way, 'blue'), colored(urban_zones, 'yellow'), colored(no_passing_zones, 'red')))
    zone_selection = {
        ZoneTypes.parking: parking,
        ZoneTypes.express_way: express_way,
        ZoneTypes.urban_area: urban_zones,
        ZoneTypes.no_passing: no_passing_zones
    }
    return ip_solution, zone_selection, start_index


def filter_zones(zone_desriptions, zone_type):
    filtered = []
    for zone_description in zone_desriptions:
        if zone_description.zone_type == zone_type:
            filtered.append(zone_description)
    return filtered


def find_straight_zones(graph_tour):
    straight_zones = dict()
    counter = 0
    counter_start = 0
    for index, node in enumerate(graph_tour.get_nodes()):
        if node.track_property == TrackProperties.straight:
            if counter == 0:
                counter_start = index
            counter += 1
        else:
            if counter > 0:
                counter_end = index - 1
                length = counter_end - counter_start + 1
                if length in straight_zones.keys():
                    straight_zones[length].append((counter_start-1, counter_end-1))
                else:
                    straight_zones[length] = [(counter_start-1, counter_end-1)]
            counter = 0

    return straight_zones


def get_zones_at_index(graph_tour_index, zone_selection):
    """
    return list of all zone_types that are applicable to a given index (of graph tour)
    """

    if zone_selection is None:
        return [], None

    zones = []
    start_or_end = None
    for zone_type in zone_selection.keys():
        for (start, end) in zone_selection[zone_type]:
            in_zone = False
            if start > end:  # if zone description rolls over
                if start <= graph_tour_index or graph_tour_index <= end:
                    in_zone = True
            else:
                if start <= graph_tour_index <= end:
                    in_zone = True
            if in_zone:
                zones.append(zone_type)
                if start == graph_tour_index:
                    start_or_end = (ZoneMisc.start, zone_type)
                elif end == graph_tour_index:
                    start_or_end = (ZoneMisc.end, zone_type)

    return zones, start_or_end


if __name__ == '__main__':
    # GraphModel(6, 6, generate_intersections=True, allow_gap_intersections=True, allow_adjacent_intersections=False, intersections_ip=False)
    _solution, _zone_selection, _start_index = get_zone_solution('/home/malte/PycharmProjects/circuit-creator/ip/configs/demo.txt',
                                                                 zone_descriptions=[
                                                                     ZoneDescription(ZoneTypes.express_way, min_length=3, max_length=4),
                                                                     ZoneDescription(ZoneTypes.urban_area, min_length=3, max_length=5),
                                                                     ZoneDescription(ZoneTypes.urban_area, min_length=6, max_length=10),
                                                                     ZoneDescription(ZoneTypes.no_passing, min_length=6, max_length=6),
                                                                 ])
