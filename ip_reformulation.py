from pulp import *
import numpy as np
from util import is_adjacent, add_to_list_in_dict
from termcolor import colored


class GGMSTProblem:
    """
    Grid Graph Minimum Spanning Tree
    """
    def __init__(self, width, height, extra_constraints=None, raster=False):
        # arguments
        self.width = width
        self.height = height
        self.extra_constraints = extra_constraints

        # variables
        self.nodes = []  # binary
        self.nodes_values = []  # integer
        self.nodes_intersections = []  # binary
        self.edges = []  # binary
        self.edges_values = []  # integer
        self.inverse_edges = []

        # Same variables stored in different data structures for easier access
        self.node_grid = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_values = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_intersections = [[None for y in range(self.height)] for x in range(self.width)]

        self.e_in = dict()
        self.e_in_values = dict()
        self.e_out = dict()
        self.e_out_values = dict()

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
                if self.check_bounds(x, y+1):
                    edge_to_above = LpVariable("e{}_{}to{}_{}".format(x, y, x, y+1), cat=const.LpBinary)
                    edge_from_above = LpVariable("e{}_{}to{}_{}".format(x, y+1, x, y), cat=const.LpBinary)
                    edge_to_above_value = LpVariable("e{}_{}to{}_{}(value)".format(x, y, x, y+1), 0, self.get_n(), cat=const.LpInteger)
                    edge_from_above_value = LpVariable("e{}_{}to{}_{}(value)".format(x, y+1, x, y), 0, self.get_n(), cat=const.LpInteger)
                    self.edges += [edge_to_above, edge_from_above]
                    self.edges_values += [edge_to_above_value, edge_from_above_value]
                    self.inverse_edges.append((edge_to_above, edge_from_above))
                    add_to_list_in_dict(self.e_out, (x, y), edge_to_above)
                    add_to_list_in_dict(self.e_in, (x, y), edge_from_above)
                    add_to_list_in_dict(self.e_out, (x, y+1), edge_from_above)
                    add_to_list_in_dict(self.e_in, (x, y+1), edge_to_above)
                    add_to_list_in_dict(self.e_out_values, (x, y), edge_to_above_value)
                    add_to_list_in_dict(self.e_in_values, (x, y), edge_from_above_value)
                    add_to_list_in_dict(self.e_out_values, (x, y+1), edge_from_above_value)
                    add_to_list_in_dict(self.e_in_values, (x, y+1), edge_to_above_value)
                # Init variables for edges with node to the right
                if self.check_bounds(x+1, y):
                    edge_to_right = LpVariable("e{}_{}to{}_{}".format(x, y, x+1, y), cat=const.LpBinary)
                    edge_from_right = LpVariable("e{}_{}to{}_{}".format(x+1, y, x, y), cat=const.LpBinary)
                    edge_to_right_value = LpVariable("e{}_{}to{}_{}(value)".format(x, y, x+1, y), 0, self.get_n(), cat=const.LpInteger)
                    edge_from_right_value = LpVariable("e{}_{}to{}_{}(value)".format(x+1, y, x, y), 0, self.get_n(), cat=const.LpInteger)
                    self.edges += [edge_to_right, edge_from_right]
                    self.edges_values += [edge_to_right_value, edge_from_right_value]
                    self.inverse_edges.append((edge_to_right, edge_from_right))
                    add_to_list_in_dict(self.e_out, (x, y), edge_to_right)
                    add_to_list_in_dict(self.e_in, (x, y), edge_from_right)
                    add_to_list_in_dict(self.e_out, (x+1, y), edge_from_right)
                    add_to_list_in_dict(self.e_in, (x+1, y), edge_to_right)
                    add_to_list_in_dict(self.e_out_values, (x, y), edge_to_right_value)
                    add_to_list_in_dict(self.e_in_values, (x, y), edge_from_right_value)
                    add_to_list_in_dict(self.e_out_values, (x+1, y), edge_from_right_value)
                    add_to_list_in_dict(self.e_in_values, (x+1, y), edge_to_right_value)
        
        # Init Problem
        self.problem = LpProblem("GGMSTProblem", LpMinimize)
        
        # Add constraints to the Problem
        self.add_all_constraints(raster)
        
    def get_safe(self, x, y):
        if x >= len(self.node_grid) or y >= len(self.node_grid[x]) or x < 0 or y < 0:
            return None
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

    def get_adjacent_nodes(self):
        """
        Get variables adjacent to each pixel as list
        :return: [pixel, adjacent_top, adjacent_bottom, adjacent_right, adjacent_left]
        """
        variables_list = []
        for x in range(len(self.node_grid)):
            for y in range(len(self.node_grid[x])):
                adjacent = [self.get_safe(x + _x, y + _y) for _x, _y in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
                variables = [self.node_grid[x][y]]
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
        for x in range(len(self.node_grid) - size + 1):
            for y in range(len(self.node_grid[x]) - size + 1):
                variables = []
                for _x in [idx for idx in range(size)]:
                    for _y in [idx for idx in range(size)]:
                        variable = self.node_grid[x + _x][y + _y]
                        variables.append(variable)

                variables_list.append(variables)

        return variables_list

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

    def add_no_square_cycle_constraints(self):
        """
        For each square, not all 4 pixels can be 1
        """
        squares = self.get_squares()
        for square in squares:
            self.problem += sum(square) <= 3

    def add_no_diagonal_only_constraints(self):
        """
        TODO: this constraint includes more than it was intended to! Rewrite!
        For each square, the diagonal must not be covered exclusively
        """
        squares = self.get_squares()
        for square in squares:
            self.problem += (square[0] + square[3]) / 2 <= square[1] + square[2]
            self.problem += (square[1] + square[2]) / 2 <= square[0] + square[3]
            # print("({} + {}) / 2 <= {} + {}".format(square[1], square[2], square[0], square[3]))

    def check_no_diagonal_only_constraints(self):
        """
        For each square, the diagonal must not be covered exclusively
        """
        squares = self.get_squares()
        for square in squares:
            print("({} + {}) / 2 <= {} + {} ({})".format(square[1], square[2], square[0], square[3],
                                                         (value(square[1]) + value(square[2])) / 2 <= value(square[0]) + value(square[3])))

    def add_local_adjacency_constraints(self):
        """
        For each pixel, there must be at least one adjacent pixel
        """
        adjacent_list = self.get_adjacent_nodes()
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
                column.append(self.node_grid[x][y])
            self.problem += sum(column) >= 1

        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(self.node_grid[x][y])
            self.problem += sum(row) >= 1

    def add_n_constraint(self):
        """
        All pixels combined must have a value of n
        """
        all_variables = self.get_all_nodes()
        self.problem += sum(all_variables) == self.get_n()

    def get_n(self):
        return np.ceil(self.width/2) * self.height + np.floor(self.width/2)

    def add_raster_constraint(self):
        """
        Pre determine raster shape
        """
        for x in range(0, self.width, 2):
            for y in range(0, self.height, 2):
                self.problem += self.node_grid[x][y] == 1

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
        # for x in range(self.width):
        #     for y in range(self.height):
        #         self.problem += self.node_grid_values[x][y] <= self.node_grid[x][y] * self.get_n()

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

        # Inverse edges may not be picked at the same time.
        # e.g. 0,0 -> 1,0 AND 1,0 -> 0,0 is not allowed
        # TODO: constraint should be obsolete once decreasing flow works
        for (e_forward, e_backward) in self.inverse_edges:
            self.problem += e_forward + e_backward <= 1

        # TODO: enable decreasing flow criteria
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

        # self.problem += self.node_grid_values[1][1] == 7

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
        self.add_no_square_cycle_constraints()  # needed?
        # self.add_no_diagonal_only_constraints()
        # self.add_local_adjacency_constraints()   # needed?
        # self.add_global_adjacency_constraints()
        self.add_n_constraint()
        self.add_flow_constraints()
        # if add_raster_constraint:
        #     self.add_raster_constraint()
        self.add_extra_constraints()
        # self.problem += self.node_grid[2][0] == 0

    def solve(self, _print=False, print_zeros=False):
        solution = [[0 for y in range(len(self.node_grid[x]))] for x in range(len(self.node_grid))]
        # status = self.problem.solve()
        # status = self.problem.solve(GLPK(msg = 0))  # Alternative Solver
        status = self.problem.solve(PULP_CBC_CMD(msg=0))
        if _print:
            print("{} Solution:".format(LpStatus[status]))
        for y in range(self.height-1, -1, -1):
            row = ""
            for x in range(self.width):
                solution_x_y = int(value(self.node_grid[x][y]))
                solution[x][y] = solution_x_y
                if not print_zeros:
                    solution_x_y = " " if solution_x_y == 0 else solution_x_y
                row += "{} ".format(solution_x_y)
            if _print:
                print(row)
        return solution, status+1

    def print_all_variables(self, values=True):
        print(colored("Nodes: {}, Edges: {}".format(len(self.nodes), len(self.edges)), "green"))
        print("Nodes:              {}".format(self.nodes))
        print("Node Values:        {}".format(self.nodes_values))
        # print("Node Intersections: {}".format(self.nodes_intersections))
        print("Edges:              {}".format(self.edges))
        print("Edges Values:       {}".format(self.edges_values))

        print("\nNode Grid:")
        print_grid(self.node_grid, values=values, binary=True)
        print("Node Value Grid:")
        print_grid(self.node_grid_values, values=values)
        # print("Node Intersection Grid:")
        # print_grid(self.node_grid_intersections, values=values, binary=True)

        print("Edges In dict:")
        print_dict(self.e_in, values=values, binary=True)
        print("Edge In (values) dict:")
        print_dict(self.e_in_values, values=values)
        print("Edges Out dict:")
        print_dict(self.e_out, values=values, binary=True)
        print("Edge Out (values) dict:")
        print_dict(self.e_out_values, values=values)


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


def print_grid(grid, values=False, print_zeros=True, binary=False):
    if values and value(grid[2][2]) is None:
        print("Values None, switching to names...")
        values = False
    for y in range(len(grid[0])-1, -1, -1):
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


if __name__ == '__main__':
    # p = GGMSTProblem(5, 5, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [0, 2], [1, 2], [2, 2], [2, 3], [2, 4], [1, 4], [0, 4], [0, 3]])
    # _, feasible = p.solve(_print=True)
    # p.check_no_diagonal_only_constraints()
    p = GGMSTProblem(7, 7)
    solution, status = p.solve()
    p.print_all_variables(values=True)
    print(colored("Solution {}".format(LpStatus[status-1]), "blue"))
