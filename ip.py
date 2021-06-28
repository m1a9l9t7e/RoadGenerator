from pulp import *
import numpy as np


class Problem:
    def __init__(self, width, height):
        assert (width + 1) % 2 == 0 or (height + 1) % 2 == 0
        self.width = width
        self.height = height
        self.grid = self.init_variables()
        self.problem = LpProblem("myProblem", LpMinimize)
        self.add_all_constraints()

    def init_variables(self):
        grid = [[LpVariable("{}_{}".format(x, y), 0, 1, cat=const.LpInteger) for y in range(self.height)] for x in range(self.width)]
        return grid

    def get_safe(self, x, y):
        if x >= len(self.grid) or y >= len(self.grid[x]) or x < 0 or y < 0:
            return None
        else:
            return self.grid[x][y]

    def get_all_variables(self):
        """
        Get variables adjacent to each pixel as list
        :return: [pixel, adjacent_top, adjacent_bottom, adjacent_right, adjacent_left]
        """
        variables = []
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                variables.append(self.grid[x][y])

        return variables

    def get_adjacent(self):
        """
        Get variables adjacent to each pixel as list
        :return: [pixel, adjacent_top, adjacent_bottom, adjacent_right, adjacent_left]
        """
        variables_list = []
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                adjacent = [self.get_safe(x + _x, y + _y) for _x, _y in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
                variables = [self.grid[x][y]]
                for v in adjacent:
                    if v is not None:
                        variables.append(v)
                variables_list.append(variables)

        return variables_list

    def get_squares(self, size=2):
        """
        Get variables from all 2x2 squares as list
        :return: [bottom_left, top_left, bottom_right, top_right]
        """
        variables_list = []
        for x in range(len(self.grid) - size + 1):
            for y in range(len(self.grid[x]) - size + 1):
                variables = []
                for _x in [idx for idx in range(size)]:
                    for _y in [idx for idx in range(size)]:
                        variable = self.grid[x + _x][y + _y]
                        variables.append(variable)

                variables_list.append(variables)

        return variables_list

    def add_coverage_constraints(self):
        """
        For each square, at least on pixel must be 1
        """
        squares = self.get_squares()
        for square in squares:
            self.problem += sum(square) >= 1

    def add_no_square_cycle_constraints(self):
        """
        For each square, not all 4 pixels can be 1
        """
        squares = self.get_squares()
        for square in squares:
            self.problem += sum(square) <= 3

    def add_no_diagonal_only_constraints(self):
        """
        For each square, the diagonal must not be covered exclusively
        """
        squares = self.get_squares()
        for square in squares:
            self.problem += (square[0] + square[3]) / 2 <= square[1] + square[2]
            self.problem += (square[1] + square[2]) / 2 <= square[0] + square[3]

    def add_local_adjacency_constraints(self):
        """
        For each pixel, there must be at least one adjacent pixel
        """
        adjacent_list = self.get_adjacent()
        for adjacent in adjacent_list:
            pixel = adjacent[0]
            adjacent_pixels = adjacent[1:]
            self.problem += pixel <= sum(adjacent_pixels)

    def add_global_adjacency_constraints(self):
        """
        For each row and each column, there must be at least one pixel with positive value
        """
        for x in range(self.width):
            column = []
            for y in range(self.height):
                column.append(self.grid[x][y])
            self.problem += sum(column) >= 1

        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(self.grid[x][y])
            self.problem += sum(row) >= 1

    def add_n_constraint(self):
        """
        No pixel can have a value greater than 1, all pixels combined must have a value of n
        """
        n = np.ceil(self.width/2) * self.height + np.floor(self.width/2)
        print("width: {}, height: {} => n: {}".format(self.width, self.height, n))
        all_variables = self.get_all_variables()
        self.problem += sum(all_variables) == n

        for variable in all_variables:
            self.problem += variable <= 1

    def add_raster_constraint(self):
        """
        Pre determine raster shape
        """
        for x in range(0, self.width, 2):
            for y in range(0, self.height, 2):
                self.problem += self.grid[x][y] == 1

    def add_all_constraints(self):
        self.add_coverage_constraints()  # needed?
        self.add_local_adjacency_constraints()   # needed?
        self.add_no_square_cycle_constraints()  # needed?
        self.add_no_diagonal_only_constraints()
        self.add_global_adjacency_constraints()
        self.add_raster_constraint()
        self.add_n_constraint()
        # self.problem += self.grid[1][1] == 1

    def solve(self, _print=False, print_zeros=False):
        solution = [[0 for y in range(len(self.grid[x]))] for x in range(len(self.grid))]
        status = self.problem.solve()
        # status = prob.solve(GLPK(msg = 0))  # Alternative Solver
        if _print:
            print("{} Solution:".format(LpStatus[status]))
        for y in range(self.height-1, -1, -1):
            row = ""
            for x in range(self.width):
                solution_x_y = int(value(self.grid[x][y]))
                solution[x][y] = solution_x_y
                if not print_zeros:
                    solution_x_y = " " if solution_x_y == 0 else solution_x_y
                row += "{} ".format(solution_x_y)
            if _print:
                print(row)
        return solution


def get_problem(graph_width, graph_height):
    return Problem(graph_width - 1, graph_height - 1)


if __name__ == '__main__':
    problem = Problem(5, 5)
    solution_grid = problem.solve(_print=True)
