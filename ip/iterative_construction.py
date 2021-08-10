from tqdm import tqdm
import multiprocessing as mp
import itertools
import numpy as np
import functools
from termcolor import colored
import time
from numba import jit


class Iterator:
    def __init__(self, width, height, _print=False):
        self.width = width
        self.height = height
        self._print = _print
        self.variants = []
        self.iteration_counter = 0
        self.leaf_counter = 0

    def iterate(self, depth_first=True, multi_processing=True, parallel=10000):
        start = Node(grid=create_base_grid(np.zeros((self.width, self.height), dtype=int), positive_cells=[(0, 0)], negative_cells=[]), num_positives=1)
        queue = [start]

        with tqdm(total=0) as pbar:
            while len(queue) > 0:
                if multi_processing:
                    pool = mp.Pool(mp.cpu_count())
                    queue_len = len(queue)
                    if depth_first:
                        queue, nodes = (queue[:-parallel], queue[-parallel:])
                    else:
                        queue, nodes = (queue[parallel:], queue[:parallel])

                    pbar.set_description(pretty_description(queue_len, len(nodes), self.leaf_counter, len(self.variants)))
                    full_iteration = pool.map(next_wrapper, nodes)
                    pool.close()
                    _counter = 0
                    for _next in full_iteration:
                        add_to_queue = self.unpack_next(_next)
                        queue += add_to_queue
                        self.iteration_counter += len(add_to_queue)
                        _counter += len(add_to_queue)
                    pbar.update(_counter)
                else:
                    if self.iteration_counter % 1000 == 0:
                        pbar.set_description(pretty_description(len(queue), 1, self.leaf_counter, len(self.variants)))
                    if depth_first:
                        _next = queue.pop(len(queue) - 1).get_next()
                    else:
                        _next = queue.pop(0).get_next()
                    add_to_queue = self.unpack_next(_next)
                    queue += add_to_queue
                    pbar.update(len(add_to_queue))
                    self.iteration_counter += len(add_to_queue)

                pbar.refresh()

        if self._print:
            print("Number of total iterations: {}".format(self.iteration_counter))
            print("Number of checked leafs: {}".format(self.leaf_counter))
            print("Number found variants: {}".format(len(self.variants)))

        return self.variants

    def unpack_next(self, _next):
        leaf, content = _next
        add_to_queue = []
        if leaf:
            if content is not None:
                self.variants.append(content)
            self.leaf_counter += 1
        else:
            add_to_queue = content
        return add_to_queue

    def next_wrapper(self, node):
        _next = node.get_next()
        return self.unpack_next(_next)


def next_wrapper(node):
    return node.get_next()


def create_base_grid(base_grid, positive_cells, negative_cells):
    grid = np.copy(base_grid)
    for (x, y) in positive_cells:
        grid[x][y] = 1
    for (x, y) in negative_cells:
        grid[x][y] = -1
    return grid


def count_positive_cells(grid):
    return np.sum(np.array(grid))


@jit(nopython=True)
def count_adjacent(grid, coords):
    x, y = coords

    counter = 0
    adjacent_edge = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    adjacent_corner = [[(-1, -1), (-1, 1)], [(-1, -1), (1, -1)], [(1, 1), (1, -1)], [(-1, 1), (1, 1)]]
    for index, (_x, _y) in enumerate(adjacent_edge):
        if get_safe(grid, (x + _x, y + _y)) > 0:
            for (__x, __y) in adjacent_corner[index]:
                if get_safe(grid, (x + __x, y + __y)) > 0:
                    return 2
            counter += 1

    return counter


@jit(nopython=True)
def get_safe(grid, coords):
    x, y = coords
    if x >= len(grid) or y >= len(grid[0]) or x < 0 or y < 0:
        return 0
    return grid[x][y]


@jit(nopython=True)
def make_next_grid(base_grid, combination, possibilites):
    """
    Makes new grid from given base grid, as well as a list of possible next cells and a combination of those to select.
    Note: the new cells are placed sequentially. Should two new cells be adjacent, the function will return with success=false

    :param base_grid: the grid that the new grid is based on
    :param combination: a tuple (b1, ..., bn) of booleans indicating which possible cells should be selected
    :param possibilites a list [(x1, y1), ..., (xn, yn)] of coords, representing the selectable cells
    :return: new grid, counter of new positive cells, success bool
    """
    new_positive_cells = 0
    next_grid = np.copy(base_grid)
    for index, (x, y) in enumerate(possibilites):
        if combination[index]:
            if count_adjacent(next_grid, (x, y)) != 1:
                return None, 0, False
            next_grid[x][y] = 1
            new_positive_cells += 1
        else:
            next_grid[x][y] = -1
    return next_grid, new_positive_cells, True


@functools.lru_cache()
def get_combinations(length):
    combinations = [list(i) for i in itertools.product([0, 1], repeat=length)]
    return np.array(combinations)


@functools.lru_cache()
def get_n(width, height):
    return int(np.ceil(width / 2) * np.ceil(height / 2)) * 2 - 1


class Node:
    def __init__(self, grid, num_positives, depth=0):
        self.grid = grid
        self.num_positives = num_positives
        self.depth = depth

    def get_next(self):
        # leaf node
        if self.num_positives == get_n(len(self.grid), len(self.grid[0])):
            self.export_grid()
            return True, self.grid

        possibilites = []
        indices = np.argwhere(self.grid == 0)
        for (x, y) in indices:
            if count_adjacent(self.grid, (x, y)) == 1:
                possibilites.append((x, y))

        # also leaf node, but invald
        if len(possibilites) == 0:
            return True, None

        _next = []
        combinations = get_combinations(len(possibilites))
        for combination in combinations:
            next_grid, counter, success = make_next_grid(self.grid, combination, np.array(possibilites))
            if not success:
                continue
            # if self.num_positives + counter > get_n(len(self.grid), len(self.grid[0])):
            #     print("To many cells!")
            #     continue
            _next.append(Node(next_grid, self.num_positives + counter, depth=self.depth+1))

        return False, _next

    def export_grid(self):
        self.grid = np.floor_divide(self.grid + np.ones_like(self.grid), 2)


#####################
#### BASIC UTIL #####
#####################

def print_2d(grid, print_zeros=True, highlight=None):
    height = len(grid[0])
    width = len(grid)
    for y in range(height - 1, -1, -1):
        row = ""
        for x in range(width):
            if highlight is not None:
                _x, _y = highlight
                if x == _x and y == _y:
                    row += "{} ".format(colored(grid[x][y], 'yellow'))
                    continue
            if not print_zeros:
                row += "{} ".format(" " if grid[x][y] == 0 else grid[x][y])
            else:
                row += "{} ".format(grid[x][y])
        print(row)
    print()


def pretty_description(num_queue, num_nodes, num_leafs, num_variants):
    string1 = "processing {}/{} items in queue".format(num_nodes, num_queue)
    string2 = "{} leafs checked".format(num_leafs)
    string3 = "{} variants found".format(num_variants)
    description_string = "{}, {}, {}".format(*[colored(string1, 'cyan'), colored(string2, 'yellow'), colored(string3, 'green')])
    return description_string


if __name__ == '__main__':
    print(colored("Available cores: {}\n".format(mp.cpu_count()), 'green'))
    time.sleep(0.01)
    iterator = Iterator(5, 5, _print=True)
    variants = iterator.iterate(multi_processing=False, depth_first=True, parallel=mp.cpu_count() * 5000)
    print("Variants: {}".format(len(variants)))
