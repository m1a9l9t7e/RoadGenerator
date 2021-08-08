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
        self.counter = 0

    def iterate(self, depth_first=True, multi_processing=True, parallel=10000):
        start = Node(grid=create_new_grid(np.zeros((self.width, self.height), dtype=int), positive_cells=[(0, 0)], negative_cells=[]))
        queue = [start]

        with tqdm(total=0) as pbar:
            while len(queue) > 0:
                if multi_processing:
                    pbar.set_description("{} items in queue atm".format(len(queue)))
                    pool = mp.Pool(mp.cpu_count())
                    if depth_first:
                        queue, nodes = (queue[:-parallel], queue[-parallel:])
                    else:
                        queue, nodes = (queue[parallel:], queue[:parallel])

                    full_iteration = pool.map(next_wrapper, nodes)
                    pool.close()
                    for _next in full_iteration:
                        add_to_queue = self.unpack_next(_next)
                        queue += add_to_queue
                        pbar.update(len(add_to_queue))

                else:
                    if depth_first:
                        _next = queue.pop(len(queue) - 1).get_next()
                    else:
                        _next = queue.pop(0).get_next()
                    add_to_queue = self.unpack_next(_next)
                    queue += add_to_queue
                    pbar.update(len(add_to_queue))

                pbar.refresh()

        if self._print:
            print("Number of coverage checks: {}".format(self.counter))

        return self.variants

    def unpack_next(self, _next):
        leaf, content = _next
        add_to_queue = []
        if leaf:
            if content is not None:
                self.variants.append(content)
            self.counter += 1
        else:
            add_to_queue = content
        return add_to_queue

    def next_wrapper(self, node):
        _next = node.get_next()
        return self.unpack_next(_next)


def next_wrapper(node):
    return node.get_next()


def create_new_grid(base_grid, positive_cells, negative_cells):
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
    for _x, _y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        if x + _x >= len(grid) or y + _y >= len(grid[0]) or x + _x < 0 or y + _y < 0:
            pass
        else:
            if grid[x + _x][y + _y] > 0:
                counter += 1

    return counter


@functools.lru_cache()
def get_combinations(length):
    combinations = [list(i) for i in itertools.product([0, 1], repeat=length)]
    return combinations


@functools.lru_cache()
def get_n(width, height):
    return int(np.ceil(width / 2) * np.ceil(height / 2)) * 2 - 1


def check_coverage(solution):
    """
    Checks if solution covers all blue nodes.
    Returns true if coverage is satisfied, else false.
    """
    # pad solution with zeros
    solution = np.pad(solution, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))
    # perform convolution with 2x2 ones kernel
    m, n = (2, 2)
    y, x = solution.shape
    y = y - m + 1
    x = x - m + 1
    coverage = np.zeros((y, x), dtype=int)
    for i in range(y):
        for j in range(x):
            coverage[i][j] = np.sum(solution[i:i + m, j:j + m])

    # check for zero elements
    num_uncovered = np.count_nonzero(coverage == 0)
    return num_uncovered == 0


class Node:
    def __init__(self, grid):
        self.grid = grid

    def get_next(self):
        possibilites = []
        indices = np.argwhere(self.grid == 0)
        for (x, y) in indices:
            if count_adjacent(self.grid, (x, y)) == 1:
                possibilites.append((x, y))

        # leaf node
        if len(possibilites) == 0:
            self.export_grid()
            if count_positive_cells(self.grid) == get_n(len(self.grid), len(self.grid[0])):
                if check_coverage(self.grid):
                    return True, self.grid

            return True, None

        _next = []
        combinations = get_combinations(len(possibilites))
        for combination in combinations:
            next_grid = np.copy(self.grid)
            for index, (x, y) in enumerate(possibilites):
                if combination[index]:
                    next_grid[x][y] = 1
                else:
                    next_grid[x][y] = -1
            _next.append(Node(next_grid))

        return False, _next

    def export_grid(self):
        self.grid = np.floor_divide(self.grid + np.ones_like(self.grid), 2)


def print_2d(grid, print_zeros=True):
    height = len(grid[0])
    width = len(grid)
    for y in range(height - 1, -1, -1):
        row = ""
        for x in range(width):
            if not print_zeros:
                row += "{} ".format(" " if grid[x][y] == 0 else grid[x][y])
            else:
                row += "{} ".format(grid[x][y])
        print(row)
    print()


if __name__ == '__main__':
    print(colored("Available cores: {}\n".format(mp.cpu_count()), 'green'))
    time.sleep(0.01)
    iterator = Iterator(5, 5, _print=True)
    variants = iterator.iterate(multi_processing=True, depth_first=True, parallel=mp.cpu_count() * 5000)
    print("Variants: {}".format(len(variants)))
    # count_adjacent(np.ones((3, 3), dtype=int), (0, 0))


