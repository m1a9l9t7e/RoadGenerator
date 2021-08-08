from ip.problem import Problem
from util import get_adjacent, print_2d
from tqdm import tqdm
import multiprocessing as mp
import itertools
import numpy as np


class Iterator:
    def __init__(self, width, height, _print=False):
        self.width = width
        self.height = height
        self.n = int(np.ceil(width / 2) * np.ceil(height / 2)) * 2 - 1
        self._print = _print
        self.variants = []
        self.counter = 0

    def iterate(self, multi_processing=True):
        start = Node(grid=create_new_grid([[0 for y in range(self.height)] for x in range(self.width)], positive_cells=[(0, 0)], negative_cells=[]))
        queue = [start]

        with tqdm(total=0) as pbar:
            while len(queue) > 0:
                if multi_processing:
                    pool = mp.Pool(mp.cpu_count())
                    full_iteration = pool.map(next_wrapper, queue)
                    pool.close()
                    queue = []
                    for _next in full_iteration:
                        add_to_queue = self.unpack_next(_next)
                        queue += add_to_queue
                        pbar.update(len(add_to_queue))

                else:
                    _next = queue.pop(0).get_next()
                    add_to_queue = self.unpack_next(_next)
                    queue += add_to_queue
                    pbar.update(len(add_to_queue))

                pbar.total = len(queue)
                pbar.refresh()

        if self._print:
            print("Number of IP problems solved: {}".format(self.counter))

        return self.variants

    def unpack_next(self, _next):
        leaf, content = _next
        add_to_queue = []
        if leaf:
            if count_positive_cells(content) == self.n:
                # solve ip here for verification!
                # TODO: schnelle abgedecktheit anstatt ip solve?
                problem = Problem(self.width, self.height, imitate=content)
                solution, feasible = problem.solve()
                if feasible:
                    self.variants.append(content)
                self.counter += 1
        else:
            add_to_queue = content
        return add_to_queue


def next_wrapper(node):
    return node.get_next()


def create_new_grid(base_grid, positive_cells, negative_cells):
    grid = [[base_grid[x][y] for y in range(len(base_grid[x]))] for x in range(len(base_grid))]
    for (x, y) in positive_cells:
        grid[x][y] = 1
    for (x, y) in negative_cells:
        grid[x][y] = -1
    return grid


def count_positive_cells(grid):
    return np.sum(np.array(grid))


class Node:
    def __init__(self, grid):
        self.grid = grid

    def get_next(self):
        _next = []
        possibilites = []
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == 0 and get_adjacent(self.grid, (x, y), count=True) == 1:
                    possibilites.append((x, y))

        if len(possibilites) == 0:
            return True, self.export_grid()
        combinations = [list(i) for i in itertools.product([0, 1], repeat=len(possibilites))]
        for combination in combinations:
            positive = []
            negative = []
            for index, (x, y) in enumerate(possibilites):
                if combination[index]:
                    positive.append((x, y))
                else:
                    negative.append((x, y))
            _next.append(Node(create_new_grid(self.grid, positive, negative)))

        return False, _next

    def export_grid(self):
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if self.grid[x][y] == -1:
                    self.grid[x][y] = 0

        return self.grid


if __name__ == '__main__':
    iterator = Iterator(5, 5, _print=True)
    variants = iterator.iterate(multi_processing=True)
    print("Variants: {}".format(len(variants)))
    # for variant in variants:
    #     print_2d(variant)
