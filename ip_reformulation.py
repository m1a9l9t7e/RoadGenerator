from pulp import *
import numpy as np
from util import is_adjacent, add_to_list_in_dict, time
from termcolor import colored


class GGMSTProblem:
    """
    Grid Graph Minimum Spanning Tree
    """
    def __init__(self, width, height, extra_constraints=None, n_intersections=0):
        # arguments
        self.width = width
        self.height = height
        self.extra_constraints = extra_constraints
        self.n_intersections = n_intersections
        self.n_straights = n_straights

        # variables
        self.nodes = []  # binary
        self.nodes_values = []  # integer
        self.nodes_intersections = []  # binary
        self.edges = []  # binary
        self.edges_values = []  # integer

        # Same variables stored in different data structures for easier access
        self.node_grid = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_values = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_intersections = [[None for y in range(self.height)] for x in range(self.width)]

        self.e_in = dict()
        self.e_in_values = dict()
        self.e_out = dict()
        self.e_out_values = dict()

        # extras
        self.straights = []
        self.degree_90s = []
        self.degree_180s = []

        for x in range(self.width):
            for y in range(self.height):
                # Init node variables at (x, y)
                v = LpVariable("v{}_{}".format(x, y), cat=const.LpBinary)
                v_value = LpVariable("v{}_{}(value)".format(x, y), 0, self.get_n(), cat=const.LpInteger)
                v_intersection = LpVariable("v{}_{}(intersection)".format(x, y), cat=const.LpBinary)
                self.nodes.append(v)
                self.nodes_values.append(v_value)
                self.nodes_intersections.append(v_intersection)
                self.node_grid[x][y] = v
                self.node_grid_values[x][y] = v_value
                self.node_grid_intersections[x][y] = v_intersection
                # Init variables for edges with node above
                if self.check_bounds(x, y + 1):
                    edge_to_above = LpVariable("e{}_{}to{}_{}".format(x, y, x, y + 1), cat=const.LpBinary)
                    edge_from_above = LpVariable("e{}_{}to{}_{}".format(x, y + 1, x, y), cat=const.LpBinary)
                    edge_to_above_value = LpVariable("e{}_{}to{}_{}(value)".format(x, y, x, y + 1), 0, self.get_n(), cat=const.LpInteger)
                    edge_from_above_value = LpVariable("e{}_{}to{}_{}(value)".format(x, y + 1, x, y), 0, self.get_n(), cat=const.LpInteger)
                    self.edges += [edge_to_above, edge_from_above]
                    self.edges_values += [edge_to_above_value, edge_from_above_value]
                    add_to_list_in_dict(self.e_out, (x, y), edge_to_above)
                    add_to_list_in_dict(self.e_in, (x, y), edge_from_above)
                    add_to_list_in_dict(self.e_out, (x, y + 1), edge_from_above)
                    add_to_list_in_dict(self.e_in, (x, y + 1), edge_to_above)
                    add_to_list_in_dict(self.e_out_values, (x, y), edge_to_above_value)
                    add_to_list_in_dict(self.e_in_values, (x, y), edge_from_above_value)
                    add_to_list_in_dict(self.e_out_values, (x, y + 1), edge_from_above_value)
                    add_to_list_in_dict(self.e_in_values, (x, y + 1), edge_to_above_value)
                # Init variables for edges with node to the right
                if self.check_bounds(x + 1, y):
                    edge_to_right = LpVariable("e{}_{}to{}_{}".format(x, y, x + 1, y), cat=const.LpBinary)
                    edge_from_right = LpVariable("e{}_{}to{}_{}".format(x + 1, y, x, y), cat=const.LpBinary)
                    edge_to_right_value = LpVariable("e{}_{}to{}_{}(value)".format(x, y, x + 1, y), 0, self.get_n(), cat=const.LpInteger)
                    edge_from_right_value = LpVariable("e{}_{}to{}_{}(value)".format(x + 1, y, x, y), 0, self.get_n(), cat=const.LpInteger)
                    self.edges += [edge_to_right, edge_from_right]
                    self.edges_values += [edge_to_right_value, edge_from_right_value]
                    add_to_list_in_dict(self.e_out, (x, y), edge_to_right)
                    add_to_list_in_dict(self.e_in, (x, y), edge_from_right)
                    add_to_list_in_dict(self.e_out, (x + 1, y), edge_from_right)
                    add_to_list_in_dict(self.e_in, (x + 1, y), edge_to_right)
                    add_to_list_in_dict(self.e_out_values, (x, y), edge_to_right_value)
                    add_to_list_in_dict(self.e_in_values, (x, y), edge_from_right_value)
                    add_to_list_in_dict(self.e_out_values, (x + 1, y), edge_from_right_value)
                    add_to_list_in_dict(self.e_in_values, (x + 1, y), edge_to_right_value)

        # Init Problem
        self.problem = LpProblem("GGMSTProblem", LpMinimize)

        # Add constraints to the Problem
        self.add_all_constraints()

    def get_safe(self, x, y, nonexistent=None):
        """
        :param nonexistent: What to return if requested cell does not exist 
        """
        if x >= len(self.node_grid) or y >= len(self.node_grid[x]) or x < 0 or y < 0:
            return nonexistent
        else:
            return self.node_grid[x][y]

    def check_bounds(self, x, y):
        if x >= len(self.node_grid) or y >= len(self.node_grid[x]) or x < 0 or y < 0:
            return False
        else:
            return True

    def get_all_nodes(self):
        """
        Get variables adjacent to each pixel as list
        :return: [pixel, adjacent_top, adjacent_bottom, adjacent_right, adjacent_left]
        """
        variables = []
        for x in range(len(self.node_grid)):
            for y in range(len(self.node_grid[x])):
                variables.append(self.node_grid[x][y])

        return variables

    def get_squares(self, size=2):
        """
        Get variables from all 2x2 squares as list
        :return: [bottom_left, top_left, bottom_right, top_right]
        """
        variables_list = []
        for x in range(len(self.node_grid) - size + 1):
            for y in range(len(self.node_grid[x]) - size + 1):
                variables = []
                for _x in [idx for idx in range(size)]:
                    for _y in [idx for idx in range(size)]:
                        variable = self.node_grid[x + _x][y + _y]
                        variables.append(variable)

                variables_list.append(variables)

        return variables_list

    def add_no_square_cycle_constraints(self):
        """
        For each square, not all 4 pixels can be 1
        """
        squares = self.get_squares()
        for square in squares:
            self.problem += sum(square) <= 3

    def add_90_degree_turn_constraint(self, n):
        # TODO
        pass

    def add_180_degree_turn_constraint(self, n):
        for x in range(self.width):
            for y in range(self.height):
                if not (x == 0 and y == 0):
                    self.problem += sum(self.e_in[(x, y)]) == self.node_grid[x][y]
                else:
                    self.problem += sum(self.e_in[(x, y)]) == 0
                    self.problem += sum(self.e_out[(x, y)]) >= 1

    def add_straights_constraints(self, length):
        indices = []
        for x in range(self.width):
            for y in range(self.height):
                indices.append((x, y))
        for (x, y) in indices:
            horizontal = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(i, 0) for i in range(length)]]
            parallel_top = [self.get_safe(x + _x, y + 1 + _y, nonexistent=0) for _x, _y in [(i, 0) for i in range(length)]]
            parallel_bottom = [self.get_safe(x + _x, y - 1 + _y, nonexistent=0) for _x, _y in [(i, 0) for i in range(length)]]
            if not any(elem is None for elem in horizontal):
                # 0 if no straights, 1 or 2 for each side
                straight_var = LpVariable("horizontal_straight{}_{}".format(x, y), cat=const.LpInteger)
                # This term will be 2 if all cells are 1. If any of the cells are zero, it will be (0-1) * 10 * zero cells + 2
                self.problem += straight_var <= 2 - sum([(cell - 1) * 10 for cell in horizontal])
                # This constraints enforces, that straight vars can be 2 if all pieces for a straight are there, else 0
                # This term will limit the variable depending on adjacent cells that are connected to the sides of the straight,
                # leading to a curve on one or both sides. The value shall be limited to 2, 1, 0, depending on if there are
                # adjacent cells on both, one or none of the sides
                # For any combination of elements from left/right. If one is true, value can be at most 1, if two are true, value must be zero
                for i in range(length):
                    for j in range(length):
                        self.problem += straight_var <= 2 - parallel_top[i] + parallel_bottom[j]
                self.straights.append(straight_var)
            vertical = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(0, i) for i in range(length)]]
            parallel_right = [self.get_safe(x + 1 + _x, y + _y, nonexistent=0) for _x, _y in [(i, 0) for i in range(length)]]
            parallel_left = [self.get_safe(x - 1 + _x, y + _y, nonexistent=0) for _x, _y in [(i, 0) for i in range(length)]]
            if not any(elem is None for elem in vertical):
                straight_var = LpVariable("vertical_straight{}_{}".format(x, y), 0, 2, cat=const.LpInteger)
                self.problem += straight_var <= sum([(cell - 1) * 10 for cell in vertical]) + 2
                for i in range(length):
                    for j in range(length):
                        self.problem += straight_var <= 2 - parallel_right[i] + parallel_left[j]
                self.straights.append(straight_var)

        self.problem += sum(self.straights) >= self.n_straights
        return self.straights

    def get_squares_fractured(self):
        """
        Get variables from all 2x2 squares as list
        :return: [bottom_left, top_left, bottom_right, top_right]
        """
        variables_list = []
        for x in range(len(self.node_grid) + 1):
            for y in range(len(self.node_grid[0]) + 1):
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

    def add_n_constraint(self):
        """
        All pixels combined must have a value of n
        """
        all_variables = self.get_all_nodes()
        self.problem += sum(all_variables) == self.get_n()

    def get_n(self):
        return np.ceil(self.width / 2) * self.height + np.floor(self.width / 2)

    def add_extra_constraints(self):
        if self.extra_constraints is not None:
            for (x, y) in self.extra_constraints:
                self.problem += self.node_grid[x][y] == 1

    def add_flow_constraints(self):
        root_node = self.node_grid[0][0]
        root_node_value = self.node_grid_values[0][0]

        # Root node must exist and have a value of n
        self.problem += root_node == 1
        self.problem += root_node_value == self.get_n()

        # Nodes can only be selected if their value is >= 1
        # In other words: if a nodes value <= 0, the node selection is zero
        for x in range(self.width):
            for y in range(self.height):
                self.problem += self.node_grid[x][y] <= self.node_grid_values[x][y]

        # Nodes can only have values > 0, if they are selected
        # In other words: if a nodes selection is zero, the nodes value is also zero
        # This constraint is technically not needed
        for x in range(self.width):
            for y in range(self.height):
                self.problem += self.node_grid_values[x][y] <= self.node_grid[x][y] * self.get_n()

        # Edges can only be selected if their value is >= 1
        # In other words: if a nodes value is zero, the selection must also be zero
        for index in range(len(self.edges)):
            self.problem += self.edges[index] <= self.edges_values[index]

        # Edges can only have values > 0, if they are selected
        # # In other words: if an edges selection is zero, the edges values is also zero
        for index in range(len(self.edges)):
            self.problem += self.edges_values[index] <= self.edges[index] * self.get_n()

        # All nodes except root node have exactly exactly one edge going in, iff they exist
        # root node has no edges going in and at least one edge going out
        for x in range(self.width):
            for y in range(self.height):
                if not (x == 0 and y == 0):
                    self.problem += sum(self.e_in[(x, y)]) == self.node_grid[x][y]
                else:
                    self.problem += sum(self.e_in[(x, y)]) == 0
                    self.problem += sum(self.e_out[(x, y)]) >= 1

        # Outgoing edges can only exist, if there is one ingoing edge
        # Since there sum(in) is either 1 or 0, sum(in) * 4 >= sum(out) holds
        for x in range(self.width):
            for y in range(self.height):
                if not (x == 0 and y == 0):
                    self.problem += sum(self.e_in[(x, y)]) * 4 >= sum(self.e_out[(x, y)])

        # Edges going out of node have value of node - 1
        # but only if edge exists? <=??
        for x in range(self.width):
            for y in range(self.height):
                for edge_value in self.e_out_values[(x, y)]:
                    self.problem += edge_value <= self.node_grid_values[x][y]
                    # self.problem += edge_value == self.node_grid_values[x][y]

        # The value of each node except root must be max(values of edges going in) - 1
        # Since we know that there is exactly one edge going in, max(values of edges going in) = sum(values of edges going in)
        for x in range(self.width):
            for y in range(self.height):
                if not (x == 0 and y == 0):
                    self.problem += self.node_grid_values[x][y] == sum(self.e_in_values[(x, y)]) - self.node_grid[x][y]

    def add_intersection_constraints(self):
        # Intersections can only be exist at selected cells.
        for index in range(len(self.nodes_intersections)):
            self.problem += self.nodes_intersections[index] <= self.nodes[index]

        # No adjacency constraints:
        indices = []
        for x in range(self.width):
            for y in range(self.height):
                indices.append((x, y))

        # Two adjacent intersection vars can not both be selected
        for (x1, y1) in indices:
            for (x2, y2) in indices:
                if is_adjacent((x1, y1), (x2, y2)):
                    self.problem += self.node_grid_intersections[x1][y1] + self.node_grid_intersections[x2][y2] <= 1

        # Number of intersections == n
        self.problem += sum(self.nodes_intersections) == self.n_intersections

        # There must an adjacent cell on both sides of a cell (left and right or top and bottom), for
        # an intersection to exist. This implies that the degree of the cell must be 2
        for (x, y) in indices:
            adjacent = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            # Exclude the following cases:
            # Right adjacent, but not left
            self.problem += self.node_grid_intersections[x][y] <= adjacent[0] * 1 + adjacent[1] * - 1 + 1
            # Left adjacent, but not right
            self.problem += self.node_grid_intersections[x][y] <= adjacent[1] * 1 + adjacent[0] * - 1 + 1
            # Top adjacent, but not bottom
            self.problem += self.node_grid_intersections[x][y] <= adjacent[2] * 1 + adjacent[3] * - 1 + 1
            # Bottom adjacent, but not top
            self.problem += self.node_grid_intersections[x][y] <= adjacent[3] * 1 + adjacent[2] * - 1 + 1
            # Not more than 2 adjacent
            self.problem += self.node_grid_intersections[x][y] <= -sum(adjacent) + 3

    def add_all_constraints(self):
        self.add_no_square_cycle_constraints()  # This is only needed for CBM performance!
        self.add_coverage_constraints()
        self.add_n_constraint()
        self.add_flow_constraints()
        self.add_extra_constraints()
        if self.n_intersections is not None:
            self.add_intersection_constraints()
        if self.n_straights is not None:
            self.add_straights_constraints(3)

    def solve(self, _print=False, print_zeros=False, intersections=False):
        solution = [[0 for y in range(len(self.node_grid[x]))] for x in range(len(self.node_grid))]
        status = self.problem.solve(PULP_CBC_CMD(msg=0))
        # status = self.problem.solve(CPLEX_PY(msg=0))

        if _print:
            print("{} Solution:".format(LpStatus[status]))
        for y in range(self.height - 1, -1, -1):
            row = ""
            for x in range(self.width):
                solution_x_y = int(value(self.node_grid[x][y]))
                if intersections:
                    solution_x_y += int(value(self.node_grid_intersections[x][y]))
                solution[x][y] = solution_x_y
                if not print_zeros:
                    solution_x_y = " " if solution_x_y == 0 else solution_x_y
                row += "{} ".format(solution_x_y)
            if _print:
                print(row)
        return solution, status + 1

    def print_all_variables(self, values=True):
        print(colored("Nodes: {}, Edges: {}".format(len(self.nodes), len(self.edges)), "green"))
        print("Nodes:              {}".format(self.nodes))
        print("Node Values:        {}".format(self.nodes_values))
        print("Node Intersections: {}".format(self.nodes_intersections))
        print("Edges:              {}".format(self.edges))
        print("Edges Values:       {}".format(self.edges_values))

        print("\nNode Grid:")
        print_grid(self.node_grid, values=values, binary=True)
        print("Node Value Grid:")
        print_grid(self.node_grid_values, values=values)
        print("Node Intersection Grid:")
        print_grid(self.node_grid_intersections, values=values, binary=True)

        print("Edges In dict:")
        print_dict(self.e_in, values=values, binary=True)
        print("Edge In (values) dict:")
        print_dict(self.e_in_values, values=values)
        print("Edges Out dict:")
        print_dict(self.e_out, values=values, binary=True)
        print("Edge Out (values) dict:")
        print_dict(self.e_out_values, values=values)

        print("Extras:")
        # print("Straights: {}".format(self.straights))
        print_list(self.straights, values=True, binary=True)
        print_list(self.straights, values=True)


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
        return solution, status + 1


def print_grid(grid, values=True, print_zeros=True, binary=False):
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
        print(row)
    print()


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
            if values and binary:
                if int(value(variable)) > 0:
                    _print = colored(str(variable), 'green')
                else:
                    _print = variable
            elif values:
                _print = int(value(variable))
            else:
                _print = variable
            list_str += "{} ".format(_print)
        _str += "{}\n".format(list_str)
    print(_str)


def print_list(_list, values=False, binary=False):
    list_str = ""
    for variable in _list:
        if values and binary:
            if int(value(variable)) > 0:
                _print = colored(str(variable), 'green')
            else:
                _print = variable
        elif values:
            _print = int(value(variable))
        else:
            _print = variable
        list_str += "{} ".format(_print)
    print(list_str)


if __name__ == '__main__':
    p = GGMSTProblem(5, 5, n_intersections=None, n_straights=0)
    start = time.time()
    solution, status = p.solve(_print=True)
    end = time.time()
    p.print_all_variables(values=True)
    print(colored("Solution {}, Time elapsed: {:.2f}s".format(LpStatus[status - 1], end - start), "blue"))
