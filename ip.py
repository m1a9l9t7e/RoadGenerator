from pulp import *
import numpy as np
from util import is_adjacent


class GGMSTProblem:
    """
    Grid Graph Minimum Spanning Tree
    """
    def __init__(self, width, height, extra_constraints=None, raster=False):
        # assert width % 2 == 1 and height % 2 == 1
        self.width = width
        self.height = height
        self.grid = self.init_variables()
        self.extra_constraints = extra_constraints
        self.problem = LpProblem("GGMSTProblem", LpMinimize)
        self.add_all_constraints(raster)

    def init_variables(self):
        grid = [[LpVariable("{}_{}".format(x, y), cat=const.LpBinary) for y in range(self.height)] for x in range(self.width)]
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

    def get_squares_fractured(self):
        """
        Get variables from all 2x2 squares as list
        :return: [bottom_left, top_left, bottom_right, top_right]
        """
        variables_list = []
        for x in range(len(self.grid) + 1):
            for y in range(len(self.grid[0]) + 1):
                variables = []
                square = [self.get_safe(x + _x, y + _y) for _x, _y in [(0, 0), (0, -1), (-1, 0), (-1, -1)]]
                for v in square:
                    if v is not None:
                        variables.append(v)

                variables_list.append(variables)

        return variables_list

    def add_coverage_constraints(self):
        """
        For each square, at least on pixel must be 1
        """
        # squares = self.get_squares()
        squares = self.get_squares_fractured()
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
        all_variables = self.get_all_variables()
        self.problem += sum(all_variables) == n

    def add_raster_constraint(self):
        """
        Pre determine raster shape
        """
        for x in range(0, self.width, 2):
            for y in range(0, self.height, 2):
                self.problem += self.grid[x][y] == 1

    def add_extra_constraints(self):
        if self.extra_constraints is not None:
            for (x, y) in self.extra_constraints:
                self.problem += self.grid[x][y] == 1

    def add_90_degree_turn_constraint(self, n):
        # TODO
        pass

    def add_180_degree_turn_constraint(self, n):
        # TODO
        pass

    def add_straights_constraint(self, length, n):
        # TODO
        pass

    def add_all_constraints(self, add_raster_constraint):
        self.add_coverage_constraints()
        self.add_local_adjacency_constraints()   # needed?
        self.add_no_square_cycle_constraints()  # needed?
        self.add_no_diagonal_only_constraints()
        self.add_global_adjacency_constraints()
        self.add_n_constraint()
        self.add_extra_constraints()
        if add_raster_constraint:
            self.add_raster_constraint()
        # self.problem += self.grid[2][0] == 0

    def solve(self, _print=False, print_zeros=False):
        solution = [[0 for y in range(len(self.grid[x]))] for x in range(len(self.grid))]
        # status = self.problem.solve()
        # status = self.problem.solve(GLPK(msg = 0))  # Alternative Solver
        status = self.problem.solve(PULP_CBC_CMD(msg=0))
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
        return solution, status+1


class IntersectionProblem:
    def __init__(self, non_zero_indices, n=None, allow_adjacent=False, extra_constraints=None):
        self.non_zero_indices = non_zero_indices
        self.problem = LpProblem("IntersectionProblem", LpMinimize)
        self.variables = self.init_variables()
        self.add_all_constraints(n, allow_adjacent, extra_constraints)

    def init_variables(self):
        variables = [LpVariable("{}".format(index), cat=const.LpBinary) for index in range(len(self.non_zero_indices))]
        return variables

    def add_no_adjacency_constraints(self):
        """
        Adjacent intersection are not allowed
        """
        for i, indices1 in enumerate(self.non_zero_indices):
            for j, indices2 in enumerate(self.non_zero_indices):
                if is_adjacent(indices1, indices2):
                    self.problem += self.variables[i] + self.variables[j] <= 1

    def add_extra_constraints(self, extra_constraints):
        """
        Add intersections that must exist
        """
        for index in range(len(self.non_zero_indices)):
            if index in extra_constraints:
                self.problem += self.variables[index] == 1
            else:
                self.problem += self.variables[index] == 0

    def add_diagonal_constraint(self, n):
        # TODO
        pass  # This could also be formulated as an objective

    def add_no_stub_intersection_constraint(self, max_n):
        # TODO
        pass

    def add_all_constraints(self, n, allow_adjacent, extra_constraints):
        if not allow_adjacent:
            self.add_no_adjacency_constraints()
        if extra_constraints is not None:
            self.add_extra_constraints(extra_constraints)
        if n is not None:
            self.problem += sum(self.variables) == n

    def solve(self, _print=False):
        status = self.problem.solve(PULP_CBC_CMD(msg=0))
        solution = [value(variable) for variable in self.variables]
        if _print:
            print("{} Solution: {}".format(LpStatus[status], solution))
        return solution, status+1


if __name__ == '__main__':
    # p = Problem(3, 3, [[0, 1], [1, 0]])
    p = GGMSTProblem(3, 3)
    solution_grid, success = p.solve(_print=False)
