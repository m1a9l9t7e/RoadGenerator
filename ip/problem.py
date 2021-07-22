from pulp import *
import numpy as np
from ip.ip_util import TrackProperties, grid_as_str, dict_as_str, list_as_str, export_grid, export_dict, export_list, QuantityConstraint, ConditionTypes, sort_quantity_constraints, list_grid_as_str, \
    export_list_grid
from util import is_adjacent, add_to_list_in_dict, time, Capturing
from termcolor import colored


class Problem:
    """
    Grid Graph Minimum Spanning Tree
    """
    def __init__(self, width, height, iteration_constraints=None, quantity_constraints=[]):
        # arguments
        self.width = width
        self.height = height
        self.iteration_constraints = iteration_constraints
        self.quantity_constraints = sort_quantity_constraints(quantity_constraints)

        # variables
        self.nodes = []  # binary
        self.nodes_values = []  # integer
        self.edges = []  # binary
        self.edges_values = []  # integer

        # Same variables stored in different data structures for easier access
        self.node_grid = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_values = [[None for y in range(self.height)] for x in range(self.width)]

        self.e_in = dict()
        self.e_in_values = dict()
        self.e_out = dict()
        self.e_out_values = dict()

        # quantity constraints
        self.nodes_intersections = []
        self.nodes_90s = []
        self.nodes_180s = []
        self.nodes_straights = []
        self.node_grid_intersections = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_90s_inner = [[[] for y in range(self.height)] for x in range(self.width)]  # [bottom_right, top_right, top_left, bottom_left]
        self.node_grid_90s_outer = [[[] for y in range(self.height)] for x in range(self.width)]  # [bottom_right, top_right, top_left, bottom_left]
        self.node_grid_180s_outer = [[[] for y in range(self.height)] for x in range(self.width)]  # [right, top, left, bottom]
        self.node_grid_180s_inner = [[[] for y in range(self.height)] for x in range(self.width)]  # [right_top, right_bottom, top_right, top_left]
        self.node_grid_straights_horizontal = [[[] for y in range(self.height)] for x in range(self.width)]  # [bottom, top]
        self.node_grid_straights_vertical = [[[] for y in range(self.height)] for x in range(self.width)]  # [left, right]

        for x in range(self.width):
            for y in range(self.height):
                # Init node variables at (x, y)
                v = LpVariable("v{}_{}".format(x, y), cat=const.LpBinary)
                v_value = LpVariable("v{}_{}(value)".format(x, y), 0, self.get_n(), cat=const.LpInteger)
                self.nodes.append(v)
                self.nodes_values.append(v_value)
                self.node_grid[x][y] = v
                self.node_grid_values[x][y] = v_value
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

    ################################
    ######## CORE METHODS ##########
    ################################

    def add_all_constraints(self):
        self.add_no_square_cycle_constraints()  # This is only needed for CBM performance!
        self.add_coverage_constraints()
        self.add_n_constraint()
        self.add_flow_constraints()
        self.add_iteration_constraints()
        for quantity_constraint in self.quantity_constraints:
            _type = quantity_constraint.property_type
            if _type == TrackProperties.intersection:
                variables = self.add_intersection_constraints()
            elif _type == TrackProperties.straight:
                variables = self.add_straights_constraints(3)
            elif _type == TrackProperties.turn_90:
                variables = self.get_90_degree_turn_constraints()
            elif _type == TrackProperties.turn_180:
                variables = self.get_180_degree_turn_constraints()
            else:
                raise ValueError("Track Property Type '{}' is not defined.".format(_type))

            self.problem += quantity_constraint.get_condition(variables)

    def solve(self, _print=False, print_zeros=False):
        solution = [[0 for y in range(len(self.node_grid[x]))] for x in range(len(self.node_grid))]
        try:
            with Capturing() as output:
                status = self.problem.solve(GUROBI(msg=0))
        except:
            print(colored('GUROBI IS NOT AVAILABLE. DEFAULTING TO CBM!', 'red'))
            status = self.problem.solve(PULP_CBC_CMD(msg=0))

        if status <= 0:
            return None, status + 1

        if _print:
            print("{} Solution:".format(LpStatus[status]))
        for y in range(self.height - 1, -1, -1):
            row = ""
            for x in range(self.width):
                solution_x_y = int(value(self.node_grid[x][y]))
                if self.node_grid_intersections[x][y] is not None:
                    solution_x_y += int(value(self.node_grid_intersections[x][y]))
                solution[x][y] = solution_x_y
                if not print_zeros:
                    solution_x_y = " " if solution_x_y == 0 else solution_x_y
                row += "{} ".format(solution_x_y)
            if _print:
                print(row)
        return solution, status + 1

    ################################
    ######## NON CIRCULAR ##########
    ################################

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

    ################################
    ######## CONNECTIVITY ##########
    ################################

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
        # In other words: if an edges selection is zero, the edges values is also zero
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

        # Edges going out of node have value of node
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

    ################################
    #### QUANTITY CONSTRAINTS ######
    ################################

    def add_intersection_constraints(self):
        for x in range(self.width):
            for y in range(self.height):
                v_intersection = LpVariable("v{}_{}(intersection)".format(x, y), cat=const.LpBinary)
                self.node_grid_intersections[x][y] = v_intersection
                self.nodes_intersections.append(v_intersection)

        # intersection can only exist at selected cells.
        for index in range(len(self.nodes_intersections)):
            self.problem += self.nodes_intersections[index] <= self.nodes[index]

        # No adjacency constraints:
        indices = []
        for x in range(self.width):
            for y in range(self.height):
                indices.append((x, y))

        # Two adjacent intersections can not both be selected
        for (x1, y1) in indices:
            for (x2, y2) in indices:
                if is_adjacent((x1, y1), (x2, y2)):
                    self.problem += self.node_grid_intersections[x1][y1] + self.node_grid_intersections[x2][y2] <= 1

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

        # Add quantity condition
        return self.nodes_intersections

    def add_turn_constraints(self):
        indices = self.get_grid_indices()

        # Inner 90s
        for (x, y) in indices:
            self.node_grid_90s_inner[x][y] = list()
            adjacent = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(0, -1), (1, 0), (0, 1), (-1, 0)]]
            adjacent_intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, -1), (1, 0), (0, 1), (-1, 0)]]
            for (idx1, idx2) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                v_90 = LpVariable("v{}_{}(90_inner_{}_{})".format(x, y, idx1, idx2), cat=const.LpBinary)
                self.node_grid_90s_inner[x][y].append(v_90)
                self.nodes_90s.append(v_90)
                # 90 degree turns can only exist at selected cells.
                self.problem += v_90 <= self.node_grid[x][y]
                # the two cells adjacent to the corner must both exist
                self.problem += v_90 <= (adjacent[idx1] + adjacent[idx2]) / 2
                # None of the two adjacent cells may be intersections
                self.problem += v_90 <= 1 - (adjacent_intersections[idx1] + adjacent_intersections[idx2]) / 2
                # The reverse must also hold
                self.problem += adjacent[idx1] + adjacent[idx2] - (adjacent_intersections[idx1] + adjacent_intersections[idx2]) <= 1 + v_90 + 3 * (1 - self.node_grid[x][y])

        # outer 90s
        for (x, y) in indices:
            self.node_grid_90s_outer[x][y] = list()
            adjacent = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(0, -1), (1, 0), (0, 1), (-1, 0)]]
            for (idx1, idx2) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                v_90 = LpVariable("v{}_{}(90_outer_{}_{})".format(x, y, idx1, idx2), cat=const.LpBinary)
                self.node_grid_90s_outer[x][y].append(v_90)
                self.nodes_90s.append(v_90)
                # 90 degree turns can only exist at selected cells.
                self.problem += v_90 <= self.node_grid[x][y]
                # the two cells adjacent to the corner must both not exist
                self.problem += v_90 <= 1 - (adjacent[idx1] + adjacent[idx2]) / 2
                # The reverse must also hold)
                self.problem += adjacent[idx1] + adjacent[idx2] >= self.node_grid[x][y] - v_90

        # Two inner 90s result in a 180
        # Adjacent inner 90s between corners of adjacent cells
        # Corners anti-clockwise starting at bottom right
        # Vertical -> 0 and 1, 2 and 3, Horizontal -> 0 and 3, 1 and 2
        inner_180s = []
        for (x, y) in indices:
            _180s = []
            v_corners = self.node_grid_90s_inner[x][y]
            v_right_corners = self.get_safe(x + 2, y, grid=self.node_grid_90s_inner)
            if v_right_corners is not None:
                top_180 = LpVariable("v{}_{}(180_horizontal_top)".format(x, y), cat=const.LpBinary)
                self.problem += top_180 <= (v_corners[1] + v_right_corners[2]) / 2
                bottom_180 = LpVariable("v{}_{}(180_horizontal_bottom)".format(x, y), cat=const.LpBinary)
                self.problem += bottom_180 <= (v_corners[0] + v_right_corners[3]) / 2
                # Reverse must hold
                self.problem += v_corners[1] + v_right_corners[2] <= 1 + top_180
                self.problem += v_corners[0] + v_right_corners[3] <= 1 + bottom_180
                _180s += [top_180, bottom_180]
            else:
                _180s += [0, 0]

            v_top_list = self.get_safe(x, y + 2, grid=self.node_grid_90s_inner)
            if v_top_list is not None:
                right_180 = LpVariable("v{}_{}(180_vertical_right)".format(x, y), cat=const.LpBinary)
                self.problem += right_180 <= (v_corners[1] + v_top_list[0]) / 2
                left_180 = LpVariable("v{}_{}(180_vertical_left)".format(x, y), cat=const.LpBinary)
                self.problem += left_180 <= (v_corners[2] + v_top_list[3]) / 2
                # Reverse must hold
                self.problem += v_corners[1] + v_top_list[0] <= 1 + right_180
                self.problem += v_corners[2] + v_top_list[3] <= 1 + left_180
                _180s += [right_180, left_180]
            else:
                _180s += [0, 0]

            self.node_grid_180s_inner[x][y] = _180s
            self.nodes_180s += _180s
            inner_180s += _180s

        # Two Outer 90s from the same node also from a 180
        # Corners anti-clockwise starting at bottom right
        # 0 and 1, 1 and 2, 2 and 3, 3 and 0
        outer_180s = []
        for (x, y) in indices:
            _180s = []
            v_corners = self.node_grid_90s_outer[x][y]
            for (idx1, idx2) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                v_180 = LpVariable("v{}_{}(180_outer_{}_{})".format(x, y, idx1, idx2), cat=const.LpBinary)
                # 180 can only be positive if both corners are positive
                self.problem += v_180 <= (v_corners[idx1] + v_corners[idx2]) / 2
                # Reverse must hold
                self.problem += v_corners[idx1] + v_corners[idx2] <= 1 + v_180
                _180s.append(v_180)
            self.node_grid_180s_outer[x][y] = _180s
            self.nodes_180s += _180s
            outer_180s += _180s

        # One positive 180 should cancel out two 90s, therefore each 180 is weighted with -2 and appended to the 90s list
        for idx, v_180 in enumerate(inner_180s + outer_180s):
            weighted_negative = LpVariable("90s_negative_{})".format(idx), cat=const.LpInteger)
            self.problem += weighted_negative == -2 * v_180
            self.nodes_90s.append(weighted_negative)

        return self.nodes_90s, self.nodes_180s

    def add_straights_constraints(self, length):
        """
        The Idea is to represent a straight sequence of "length" cells as a new variable.
        This variable is either 2, 1 or 0.
        0: One of the cells is not selected OR there are adjacent cells on BOTH sides of the sequence
        1: All cells are selected and there are adjacent cells one ONE side of the sequence
        2: All cells are selected and there are no adjacent cells on either side of the sequence
        The value of the variable represents the number of resulting straights.
        :param length:
        :param n_straights:
        :return:
        """
        indices = self.get_grid_indices()
        for (x, y) in indices:
            horizontal = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(i, 0) for i in range(length)]]
            intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i, 0) for i in range(length)]]
            parallel_top = [self.get_safe(x + _x, y + 1 + _y, nonexistent=0) for _x, _y in [(i, 0) for i in range(length)]]
            parallel_bottom = [self.get_safe(x + _x, y - 1 + _y, nonexistent=0) for _x, _y in [(i, 0) for i in range(length)]]
            if not any(elem is None for elem in horizontal):
                bottom_straight = self.single_straight_constraint(x, y, horizontal, intersections, parallel_bottom, 'horizontal', 'bottom')
                self.nodes_straights.append(bottom_straight)
                top_straight = self.single_straight_constraint(x, y, horizontal, intersections, parallel_top, 'horizontal', 'top')
                self.nodes_straights.append(top_straight)
                self.node_grid_straights_horizontal[x][y] = [bottom_straight, top_straight]
            else:
                self.node_grid_straights_horizontal[x][y] = [0, 0]

            vertical = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(0, i) for i in range(length)]]
            intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, i) for i in range(length)]]
            parallel_right = [self.get_safe(x + 1 + _x, y + _y, nonexistent=0) for _x, _y in [(0, i) for i in range(length)]]
            parallel_left = [self.get_safe(x - 1 + _x, y + _y, nonexistent=0) for _x, _y in [(0, i) for i in range(length)]]
            if not any(elem is None for elem in vertical):
                left_straight = self.single_straight_constraint(x, y, vertical, intersections, parallel_left, 'vertical', 'left')
                self.nodes_straights.append(left_straight)
                right_straight = self.single_straight_constraint(x, y, vertical, intersections, parallel_right, 'vertical', 'right')
                self.nodes_straights.append(right_straight)
                self.node_grid_straights_vertical[x][y] = [left_straight, right_straight]
            else:
                self.node_grid_straights_vertical[x][y] = [0, 0]

        return self.nodes_straights

    def single_straight_constraint(self, x, y, cells, intersections, parallel, direction, side=""):
        # This variable can be 1 or 0 if a straight is possible. If a straight is not possible, this variable must be zero.
        straight_var = LpVariable("{}_straight_{}_n{}_x{}_y{}".format(direction, side, len(cells), x, y), cat=const.LpBinary)
        # --> If any core cell is 0, the straight var must also be zero
        self.problem += straight_var <= sum(cells) / len(cells)
        # --> If any of the parallel cells are positive, the straight var must be zero
        self.problem += straight_var <= 1 - sum(parallel) / len(parallel)
        # --> There must be an edge going in or out of the first and last cell in the direction of the straight.
        if direction == 'horizontal':
            edges_left = self.get_edges_between_safe((x - 1, y), (x, y))
            edges_right = self.get_edges_between_safe((x + len(cells) - 1, y), (x + len(cells), y))
            edges_in, edges_out = (edges_left, edges_right)
        elif direction == 'vertical':
            edges_bottom = self.get_edges_between_safe((x, y - 1), (x, y))
            edges_top = self.get_edges_between_safe((x, y + len(cells) - 1), (x, y + len(cells)))
            edges_in, edges_out = (edges_bottom, edges_top)
        else:
            raise ValueError('Direction must be horizontal or vertical. Received: {}'.format(direction))

        self.problem += straight_var <= (sum(edges_in) + sum(edges_out)) / 2
        # --> If any core cell is an intersection, the straight var must also be zero
        self.problem += straight_var <= 1 - sum(intersections) / len(intersections)

        # Reverse direction must also hold: var is 0 => at least one condition not satisfied
        self.problem += 1 - straight_var <= len(cells) - sum(cells) + sum(parallel) + 2 - (sum(edges_in) + sum(edges_out)) + sum(intersections)
        return straight_var

    ################################
    ########## ITERATION ###########
    ################################

    def add_iteration_constraints(self):
        if self.iteration_constraints is not None:
            for (x, y) in self.iteration_constraints:
                self.problem += self.node_grid[x][y] == 1

    ################################
    ############ UTIL ##############
    ################################

    def print_all_variables(self, values=True):
        print(colored("Nodes: {}, Edges: {}".format(len(self.nodes), len(self.edges)), "green"))
        # print("\nNode Grid:")
        # print(grid_as_str(self.node_grid, values=values, binary=True))
        # print("Node Value Grid:")
        # print(grid_as_str(self.node_grid_values, values=values))
        #
        # print("Edges In dict:")
        # print(dict_as_str(self.e_in, values=values, binary=True))
        # print("Edge In (values) dict:")
        # print(dict_as_str(self.e_in_values, values=values))
        # print("Edges Out dict:")
        # print(dict_as_str(self.e_out, values=values, binary=True))
        # print("Edge Out (values) dict:")
        # print(dict_as_str(self.e_out_values, values=values))

        if len(self.nodes_intersections) > 0:
            print("Node Intersection Grid:")
            grid_as_str(self.node_grid_intersections, values=values, binary=True)

        if len(self.nodes_straights) > 0:
            print("Horizontal Straights Grid:")
            print(list_grid_as_str(self.node_grid_straights_horizontal, values=values, binary=False))
            print("Vertical Straights Grid:")
            print(list_grid_as_str(self.node_grid_straights_vertical, values=values, binary=False))

        if len(self.nodes_90s) > 0:
            print("Inner 90s Grid:")
            print(list_grid_as_str(self.node_grid_90s_inner, values=values, binary=False))
            print("Outer 90s Grid:")
            print(list_grid_as_str(self.node_grid_90s_outer, values=values, binary=False))

        if len(self.nodes_180s) > 0:
            print("Outer 180s Grid:")
            print(list_grid_as_str(self.node_grid_180s_outer, values=values, binary=True))
            print("Inner 180s Grid:")
            print(list_grid_as_str(self.node_grid_180s_inner, values=values, binary=True))

    def export_variables(self, values=True):
        # export base variables
        _dict = {
            'node_grid': export_grid(self.node_grid),
            'node_grid_values': export_grid(self.node_grid_values),
            'edges_in': export_dict(self.e_in, save_name=True),
            'edges_in_values': export_dict(self.e_in_values, save_name=True),
            'edges_out': export_dict(self.e_out, save_name=True),
            'edges_out_values': export_dict(self.e_out_values, save_name=True),
        }

        # export variables from quantity constraints
        for quantity_constraint in self.quantity_constraints:
            _type = quantity_constraint.property_type
            if _type == TrackProperties.intersection:
                _dict['intersections'] = export_grid(self.node_grid_intersections)
            elif _type == TrackProperties.straight:
                _dict['horizontal_straights'] = export_list_grid(self.node_grid_straights_horizontal)
                _dict['vertical_straights'] = export_list_grid(self.node_grid_straights_vertical)
            elif _type == TrackProperties.turn_90:
                _dict['90s_inner'] = export_list_grid(self.node_grid_90s_inner)
                _dict['90s_outer'] = export_list_grid(self.node_grid_90s_outer)
            elif _type == TrackProperties.turn_180:
                _dict['180s_inner'] = export_list_grid(self.node_grid_180s_inner)
                _dict['180s_outer'] = export_list_grid(self.node_grid_180s_outer)
            else:
                raise ValueError("Track Property Type '{}' is not defined.".format(_type))

        return _dict

    def get_safe(self, x, y, nonexistent=None, grid=None):
        """
        :param nonexistent: What to return if requested cell does not exist
        :param grid: Which grid to get the result from. If none, use self.node_grid
        """
        if grid is None:
            grid = self.node_grid

        if x >= len(grid) or y >= len(grid[x]) or x < 0 or y < 0:
            return nonexistent
        else:
            if grid[x][y] is None:
                return nonexistent
            else:
                return grid[x][y]

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

    def get_edges_between_safe(self, coords1, coords2):
        if self.get_safe(*coords1) is None or self.get_safe(*coords2) is None:
            return [0, 0]
        else:
            return self.get_edges_between(coords1, coords2)

    def get_edges_between(self, coords1, coords2):
        edge1 = set.intersection(set(self.e_out[coords1]), set(self.e_in[coords2]))
        edge2 = set.intersection(set(self.e_out[coords2]), set(self.e_in[coords1]))
        return list(edge1) + list(edge2)

    def print_selected_straights(self, length):
        print("Selected straights:")
        counter = 0
        indices = []
        for x in range(self.width):
            for y in range(self.height):
                indices.append((x, y))
        for (x, y) in indices:
            horizontal = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(i, 0) for i in range(length)]]
            if not any(elem is None for elem in horizontal):
                if value(self.nodes_straights[counter]) > 0:
                    print("{} {}".format(colored("({}|{})".format(x, y), 'blue'), colored('horizontal_bottom', 'yellow')))
                counter += 1
                if value(self.nodes_straights[counter]) > 0:
                    print("{} {}".format(colored("({}|{})".format(x, y), 'blue'), colored('horizontal_top', 'yellow')))
                counter += 1
            vertical = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(0, i) for i in range(length)]]
            if not any(elem is None for elem in vertical):
                if value(self.nodes_straights[counter]) > 0:
                    print("{} {}".format(colored("({}|{})".format(x, y), 'blue'), colored('vertical_left', 'yellow')))
                counter += 1
                if value(self.nodes_straights[counter]) > 0:
                    print("{} {}".format(colored("({}|{})".format(x, y), 'blue'), colored('vertical_right', 'yellow')))
                counter += 1

    def get_grid_indices(self):
        indices = []
        for x in range(self.width):
            for y in range(self.height):
                indices.append((x, y))
        return indices

    def get_90_degree_turn_constraints(self):
        if len(self.nodes_90s) == 0:
            self.add_turn_constraints()

        return self.nodes_90s

    def get_180_degree_turn_constraints(self):
        if len(self.nodes_180s) == 0:
            self.add_turn_constraints()

        return self.nodes_180s

    def get_stats(self):
        num_intersections = 0
        num_90s = "?"
        num_180s = "?"
        num_straights = "?"

        if len(self.nodes_intersections) > 0:
            num_intersections = sum([int(value(v)) for v in self.nodes_intersections])

        if len(self.nodes_90s) > 0:
            num_90s = sum([int(value(v)) for v in self.nodes_90s])

        if len(self.nodes_180s) > 0:
            num_180s = sum([int(value(v)) for v in self.nodes_180s])

        if len(self.nodes_straights) > 0:
            num_straights = sum([int(value(v)) for v in self.nodes_straights])

        intersection_str = colored('{} intersections'.format(num_intersections), 'yellow')
        _90_str = colored('{} 90 degree turns'.format(num_90s), 'blue')
        _180_str = colored('{} 180 degree turns'.format(num_180s), 'magenta')
        straights_str = colored('{} straights of length 3'.format(num_straights), 'cyan')
        print("Solution has {}, {}, {} and {}".format(intersection_str, _90_str, _180_str, straights_str))


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


if __name__ == '__main__':
    quantity_constraints = [
        QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, 0),
        QuantityConstraint(TrackProperties.straight, ConditionTypes.more_or_equals, 0),
        QuantityConstraint(TrackProperties.turn_180, ConditionTypes.more_or_equals, 0),
        QuantityConstraint(TrackProperties.turn_90, ConditionTypes.more_or_equals, 4)
    ]
    p = Problem(5, 5, quantity_constraints=quantity_constraints, iteration_constraints=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 4], [2, 0], [2, 2], [2, 3], [2, 4], [3, 0], [3, 3], [4, 0], [4, 2], [4, 3], [4, 4]])
    start = time.time()
    solution, status = p.solve(_print=True)
    end = time.time()
    p.print_all_variables()
    # p.print_selected_straights(3)
    # p.add_180_degree_turn_constraint_debug()
    print(colored("Solution {}, Time elapsed: {:.2f}s".format(LpStatus[status - 1], end - start), "green" if status > 1 else "red"))
    p.get_stats()
