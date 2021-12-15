from pulp import *
import numpy as np
from ip.ip_util import grid_as_str, dict_as_str, list_as_str, export_grid, export_dict, export_list, QuantityConstraint, ConditionTypes, sort_quantity_constraints, \
    list_grid_as_str, export_list_grid, export_list_dict_grid, QuantityConstraintStraight, SolutionEntries, minimize_objective, export_variable
from util import is_adjacent, add_to_list_in_dict, time, Capturing, TrackProperties, print_2d
from termcolor import colored


class Problem:
    """
    Grid Graph Minimum Spanning Tree
    """
    def __init__(self, width, height, quantity_constraints=[], iteration_constraints=None, prohibition_constraints=None, full_prohibition_constraints=None, imitate=None,
                 allow_gap_intersections=False, allow_adjacent_intersections=False):
        # arguments
        self.width = width
        self.height = height
        self.allow_gap_intersections = allow_gap_intersections
        self.allow_adjacent_intersections = allow_adjacent_intersections
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
        self.nodes_straights = dict()
        self.node_grid_intersections = [[None for y in range(self.height)] for x in range(self.width)]
        self.node_grid_90s_inner = [[[] for y in range(self.height)] for x in range(self.width)]  # [bottom_right, top_right, top_left, bottom_left]
        self.node_grid_90s_outer = [[[] for y in range(self.height)] for x in range(self.width)]  # [bottom_right, top_right, top_left, bottom_left]
        self.node_grid_180s_outer = [[[] for y in range(self.height)] for x in range(self.width)]  # [right, top, left, bottom]
        self.node_grid_180s_inner = [[[] for y in range(self.height)] for x in range(self.width)]  # [right_top, right_bottom, top_right, top_left]
        self.node_grid_straights_horizontal = [[dict() for y in range(self.height)] for x in range(self.width)]  # [bottom, top]
        self.node_grid_straights_vertical = [[dict() for y in range(self.height)] for x in range(self.width)]  # [left, right]

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
        if minimize_objective(quantity_constraints):
            self.problem = LpProblem("Problem", LpMinimize)
        else:
            self.problem = LpProblem("Problem", LpMaximize)

        # Add constraints to the Problem
        self.add_all_constraints()

        # Iteration constraints
        if iteration_constraints is not None:
            self.add_iteration_constraints(iteration_constraints)

        # imitation
        if imitate is not None:
            self.add_imitation_constraints(imitate)

        # prohibition
        if prohibition_constraints is not None:
            self.add_prohibition_constraints(prohibition_constraints)

        # prohibition
        if full_prohibition_constraints is not None:
            self.add_prohibition_constraints_with_intersections(full_prohibition_constraints)

    ################################
    ######## CORE METHODS ##########
    ################################

    def add_all_constraints(self):
        self.add_no_square_cycle_constraints()  # This is only needed for CBM performance!
        self.add_coverage_constraints()
        self.add_n_constraint()
        self.add_flow_constraints()
        for quantity_constraint in self.quantity_constraints:
            _type = quantity_constraint.property_type
            if _type == TrackProperties.intersection:
                variables = self.add_intersection_constraints()
            elif _type == TrackProperties.straight:
                variables = self.add_straights_constraints(quantity_constraint.length)
            elif _type == TrackProperties.turn_90:
                variables = self.get_90_degree_turn_constraints()
            elif _type == TrackProperties.turn_180:
                variables = self.get_180_degree_turn_constraints()
            else:
                raise ValueError("Track Property Type '{}' is not defined.".format(_type))

            if quantity_constraint.objective is not None:
                self.problem += sum(variables)
            else:
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
                    if solution_x_y == 0:
                        solution_x_y += SolutionEntries.negative_and_intersection * int(value(self.node_grid_intersections[x][y]))
                    else:
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
        return ((self.width + 1) * (self.height + 1) - 4) / 2 + 1

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
        # Note: technically not needed?
        # for x in range(self.width):
        #     for y in range(self.height):
        #         self.problem += self.node_grid_values[x][y] <= self.node_grid[x][y] * self.get_n()

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
        # Note: technically not needed?
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

        if not self.allow_gap_intersections:
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
                    if self.allow_adjacent_intersections:
                        self.problem += self.node_grid_intersections[x1][y1] + self.node_grid_intersections[x2][y2] <= 2 - (self.node_grid[x1][y1] - self.node_grid[x2][y2])
                        self.problem += self.node_grid_intersections[x1][y1] + self.node_grid_intersections[x2][y2] <= 2 - (self.node_grid[x2][y2] - self.node_grid[x1][y1])
                    else:
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
            adjacent_intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, -1), (1, 0), (0, 1), (-1, 0)]]
            for (idx1, idx2) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                v_90 = LpVariable("v{}_{}(90_outer_{}_{})".format(x, y, idx1, idx2), cat=const.LpBinary)
                self.node_grid_90s_outer[x][y].append(v_90)
                self.nodes_90s.append(v_90)
                # 90 degree turns can only exist at selected cells.
                self.problem += v_90 <= self.node_grid[x][y]
                # the two cells adjacent to the corner must both not exist
                self.problem += v_90 <= 1 - (adjacent[idx1] + adjacent[idx2]) / 2
                # None of the two adjacent cells may be intersections
                self.problem += v_90 <= 1 - (adjacent_intersections[idx1] + adjacent_intersections[idx2]) / 2
                # The reverse must also hold)
                self.problem += adjacent[idx1] + adjacent[idx2] + adjacent_intersections[idx1] + adjacent_intersections[idx2] >= self.node_grid[x][y] - v_90

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

        # Two Outer 90s from the same node also form a 180
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
        :return:
        """
        straights = []
        indices = self.get_grid_indices()
        for (x, y) in indices:
            horizontal = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(i-1, 0) for i in range(length+1)]]

            if not any(elem is None for elem in horizontal):
                horizontal = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_top = [self.get_safe(x + _x, y + 1 + _y, nonexistent=0) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_top_intersections = [self.get_safe(x + _x, y + 1 + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_bottom = [self.get_safe(x + _x, y - 1 + _y, nonexistent=0) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_bottom_intersections = [self.get_safe(x + _x, y - 1 + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                bottom_straight = self.single_straight_constraint(x, y, horizontal, intersections, parallel_bottom, parallel_bottom_intersections, 'horizontal', 'bottom')
                straights.append(bottom_straight)
                top_straight = self.single_straight_constraint(x, y, horizontal, intersections, parallel_top, parallel_top_intersections, 'horizontal', 'top')
                straights.append(top_straight)
                self.node_grid_straights_horizontal[x][y][length] = [bottom_straight, top_straight]
            else:
                self.node_grid_straights_horizontal[x][y][length] = [0, 0]

            vertical = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(0, i-1) for i in range(length+1)]]

            if not any(elem is None for elem in vertical):
                vertical = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_right = [self.get_safe(x + 1 + _x, y + _y, nonexistent=0) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_right_intersections = [self.get_safe(x + 1 + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_left = [self.get_safe(x - 1 + _x, y + _y, nonexistent=0) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_left_intersections = [self.get_safe(x - 1 + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                left_straight = self.single_straight_constraint(x, y, vertical, intersections, parallel_left, parallel_left_intersections, 'vertical', 'left')
                straights.append(left_straight)
                right_straight = self.single_straight_constraint(x, y, vertical, intersections, parallel_right, parallel_right_intersections, 'vertical', 'right')
                straights.append(right_straight)
                self.node_grid_straights_vertical[x][y][length] = [left_straight, right_straight]
            else:
                self.node_grid_straights_vertical[x][y][length] = [0, 0]

        self.nodes_straights[length] = straights
        return straights

    def single_straight_constraint(self, x, y, cells, intersections, parallel, parallel_intersections, direction, side=""):
        """
        Set up a single straight variable independent of direction and side.
        """
        # This variable can be 1 or 0 if a straight is possible. If a straight is not possible, this variable must be zero.
        straight_var = LpVariable("{}_straight_{}_n{}_x{}_y{}".format(direction, side, len(cells) - 2, x, y), cat=const.LpBinary)
        # --> If any core cell is 0, the straight var must also be zero
        core_cells = cells[1:-1]
        self.problem += straight_var <= sum(core_cells) / len(core_cells)
        # --> If any of the parallel cells are positive, the straight var must be zero
        core_parallel = parallel[1:-1]
        self.problem += straight_var <= 1 - sum(core_parallel) / len(core_parallel)
        # --> If any core cell is an intersection, the straight var must also be zero
        core_intersections = intersections[1:-1]
        self.problem += straight_var <= 1 - sum(core_intersections) / len(core_intersections)
        # --> If any parallel cell is an intersection, the straight var must also be zero
        core_parallel_intersections = parallel_intersections[1:-1]
        self.problem += straight_var <= 1 - sum(core_parallel_intersections) / len(core_parallel_intersections)
        # --> If the start continues the straight, the straight var must be zero
        # This means the straight var may only be 1, if either the beginning cell is 0, the beginning cell is an intersection or the cell parallel to the beginning is 1
        self.problem += straight_var <= (1 - cells[0]) + intersections[0] + parallel[0] + parallel_intersections[0]
        # --> If the end continues the straight, the straight var must be zero
        # Same constraint as for start
        self.problem += straight_var <= (1 - cells[-1]) + intersections[-1] + parallel[-1] + parallel_intersections[-1]

        # Reverse direction must also hold: var is 0 => at least one condition not satisfied
        self.problem += 1 - straight_var <= len(core_cells) - sum(core_cells) + sum(core_parallel) + sum(core_intersections) \
                        + (cells[0] - intersections[0] - parallel[0] - parallel_intersections[0]) + (cells[-1] - intersections[-1] - parallel[-1] - parallel_intersections[-1])
        return straight_var

    ################################
    ########## ITERATION ###########
    ################################

    def add_iteration_constraints(self, iteration_constraints):
        for (x, y) in iteration_constraints:
            self.problem += self.node_grid[x][y] == 1

    def add_imitation_constraints(self, original_solution):
        for (x, y) in self.get_grid_indices():
            if original_solution[x][y] in [SolutionEntries.positive, SolutionEntries.positive_and_intersection]:
                self.problem += self.node_grid[x][y] == 1
            else:
                self.problem += self.node_grid[x][y] == 0
            if original_solution[x][y] in [SolutionEntries.negative_and_intersection, SolutionEntries.positive_and_intersection]:
                self.problem += self.node_grid_intersections[x][y] == 1
            else:
                self.problem += self.node_grid_intersections[x][y] == 0

    def add_prohibition_constraints(self, solutions):
        """
        Add constraints prohibiting known solutions.
        :param solutions: A list of solutions. A single solution is a list of tuples (x, y) where the corresponding cell is 1
        """
        for index, solution in enumerate(solutions):
            positive_cells = [self.node_grid[x][y] for (x, y) in solution]
            self.problem += sum(positive_cells) <= self.get_n() - 1

    def add_prohibition_constraints_with_intersections(self, full_solutions):
        """
        Add constraints prohibiting known solutions.
        Takes two params: lists of equal size and matching contents
        :param full_solutions: (cell_solution, intersection_solution), ...
        cell_solution: A list of tuples (x, y) where the corresponding cell is 1
        intersection_solution: A list of tuples (x, y) where the corresponding intersection is 1
        """
        for index, (cell_solution, intersection_solution) in enumerate(full_solutions):
            positive_cells = [self.node_grid[x][y] for (x, y) in cell_solution]
            positive_intersections = [self.node_grid_intersections[x][y] for (x, y) in intersection_solution]
            num_intersections = len(positive_intersections)

            # meta_var1: 1 if cell solution is identical, 0 else
            cells_identical = LpVariable("meta_var1_{}".format(index), cat=const.LpBinary)
            # --> Forward: if cells not identical => 0
            self.problem += cells_identical <= sum(positive_cells) / self.get_n()
            # --> Backward: 0 -> cells not identical (if cells_identical == 0, then sum(positve cells must be <= n-1!)
            self.problem += sum(positive_cells) <= self.get_n() - 1 + cells_identical

            # meta_var2a: 1 if new solution has less or equal intersections than previous solution, 0 else
            # Define var and constraints for inverse first, as this is easier:
            more_intersections = LpVariable("meta_var2a_inv{}".format(index), cat=const.LpBinary)
            # --> Forward: Less intersections than (num_intersections + 1) -> 0
            self.problem += more_intersections <= sum(self.nodes_intersections) / (num_intersections + 1)
            # --> Backward: 0 -> Less intersections than (num_intersections + 1)
            self.problem += sum(self.nodes_intersections) <= num_intersections + more_intersections * len(self.nodes_intersections)
            # Inverse
            less_or_equal_intersections = LpVariable("meta_var2a_{}".format(index), cat=const.LpBinary)
            self.problem += less_or_equal_intersections == 1 - more_intersections

            # meta_var2b: 1 if all intersections from solution are selected, 0 else
            all_previous_intersections_positive = LpVariable("meta_var2b_{}".format(index), cat=const.LpBinary)
            if len(positive_intersections) == 0:
                self.problem += all_previous_intersections_positive == 1
            else:
                # --> Forward: if not all intersections selected => 0
                self.problem += all_previous_intersections_positive <= sum(positive_intersections) / num_intersections
                # --> Backward: 0 => cells not identical (if all_previous_intersections_positive == 0, then sum(positve intersections must be <= num_intersections)
                self.problem += sum(positive_intersections) <= num_intersections - 1 + all_previous_intersections_positive

            # meta_var2: 1 if intersection solution is identical, 0 else (if meta_var2a AND meta_var2b)
            # EXPLANATION: We will list all cases of meta_var2a and meta_var2b and show that meta_var2 indicates identical intersections solutions for all:
            # Case1: Not all of the previous intersections are selected: meta_var2 = x AND 0 = 0
            # Case2: All of the previous intersections are selected, but also additional ones: meta_var2 = 0 AND 1 = 0
            # --> Here meta_var2a must be 0, because we have more intersections than previously!
            # Case3: All of the previous intersections are selected, and the number of total intersections is less or equal to before: meta_var2 = 1 AND 1 = 1
            # --> Solution must be identical
            intersections_identical = LpVariable("meta_var2_{}".format(index), cat=const.LpBinary)
            # --> Forward
            self.problem += intersections_identical <= (less_or_equal_intersections + all_previous_intersections_positive) / 2
            # --> Backward
            self.problem += intersections_identical >= less_or_equal_intersections + all_previous_intersections_positive - 1

            # meta_var3: 1 if new solution is completely identical, 0 else (if meta_var1 AND meta_var2)
            # This is not technically needed and may be replaced with constraint:
            # self.problem += cells_identical + intersections_identical <= 1

            solution_identical = LpVariable("meta_var3_{}".format(index), cat=const.LpBinary)
            # --> Forward
            self.problem += solution_identical <= (cells_identical + intersections_identical) / 2
            # --> Backward
            self.problem += solution_identical >= cells_identical + intersections_identical - 1

            # Solution may not be found again
            self.problem += solution_identical == 0

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
                _dict['horizontal_straights'] = export_list_dict_grid(self.node_grid_straights_horizontal)
                _dict['vertical_straights'] = export_list_dict_grid(self.node_grid_straights_vertical)
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

    ################################
    ## DEBUG QUANTITY CONSTRAINTS ##
    ################################

    def debug_straights_constraints(self, length, _horizontal=None, _vertical=None):
        straights = []
        node_grid_straights_horizontal = [[dict() for y in range(self.height)] for x in range(self.width)]  # [bottom, top]
        node_grid_straights_vertical = [[dict() for y in range(self.height)] for x in range(self.width)]  # [left, right]
        indices = self.get_grid_indices()
        for (x, y) in indices:
            horizontal = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(i - 1, 0) for i in range(length + 1)]]
            if _horizontal is not None:
                _top, _bottom = _horizontal[x][y]
            else:
                _top, _bottom = (None, None)

            if not any(elem is None for elem in horizontal):
                horizontal = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_top = [self.get_safe(x + _x, y + 1 + _y, nonexistent=0) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_top_intersections = [self.get_safe(x + _x, y + 1 + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_bottom = [self.get_safe(x + _x, y - 1 + _y, nonexistent=0) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                parallel_bottom_intersections = [self.get_safe(x + _x, y - 1 + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(i - 2, 0) for i in range(length + 3)]]
                bottom_straight = self.debug_single_straight_constraint(x, y, horizontal, intersections, parallel_bottom, parallel_bottom_intersections, 'horizontal', 'bottom', _eval=_bottom)
                straights.append(bottom_straight)
                top_straight = self.debug_single_straight_constraint(x, y, horizontal, intersections, parallel_top, parallel_top_intersections, 'horizontal', 'top', _eval=_top)
                straights.append(top_straight)
                node_grid_straights_horizontal[x][y][length] = [bottom_straight, top_straight]
            else:
                self.node_grid_straights_horizontal[x][y][length] = [0, 0]

            vertical = [self.get_safe(x + _x, y + _y, nonexistent=None) for _x, _y in [(0, i - 1) for i in range(length + 1)]]
            if _vertical is not None:
                _left, _right = _vertical[x][y]
            else:
                _left, _right = (None, None)


            if not any(elem is None for elem in vertical):
                vertical = [self.get_safe(x + _x, y + _y, nonexistent=0) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                intersections = [self.get_safe(x + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_right = [self.get_safe(x + 1 + _x, y + _y, nonexistent=0) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_right += [self.get_safe(x + 1 + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in
                                   [(0, i - 2) for i in range(length + 3)]]
                parallel_left = [self.get_safe(x - 1 + _x, y + _y, nonexistent=0) for _x, _y in [(0, i - 2) for i in range(length + 3)]]
                parallel_left += [self.get_safe(x - 1 + _x, y + _y, nonexistent=0, grid=self.node_grid_intersections) for _x, _y in
                                  [(0, i - 2) for i in range(length + 3)]]
                left_straight = self.debug_single_straight_constraint(x, y, vertical, intersections, parallel_left, 'vertical', 'left', _eval=_left)
                straights.append(left_straight)
                right_straight = self.debug_single_straight_constraint(x, y, vertical, intersections, parallel_right, 'vertical', 'right', _eval=_right)
                straights.append(right_straight)
                node_grid_straights_vertical[x][y][length] = [left_straight, right_straight]
            else:
                node_grid_straights_vertical[x][y][length] = [0, 0]

        # print_2d(node_grid_straights_vertical)
        print_2d(node_grid_straights_horizontal)
        # self.nodes_straights[length] = straights
        return straights

    def debug_single_straight_constraint(self, x, y, cells, intersections, parallel, parallel_intersections, direction, side="", _print=False, _eval=None, _show_positive=False):
        """
        Set up a single straight variable independent of direction and side.
        Note that parallel includes vars for parallel cells and parallel intersections as both can be true independent of each other.
        """
        # This variable can be 1 or 0 if a straight is possible. If a straight is not possible, this variable must be zero.
        straight_var = "{}_straight_{}_n{}_x{}_y{}".format(direction, side, len(cells) - 2, x, y)
        if side == 'bottom' and (x == 4 and y == 2):
            pass
        else:
            return ""
        # --> If any core cell is 0, the straight var must also be zero
        core_cells = cells[1:-1]
        if _print:
            print("{} {}".format(straight_var, '<= sum(core_cells) / len(core_cells)'))
        if _eval is not None:
            if _eval <= sum(export_list(core_cells)) / len(core_cells):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, sum(export_list(core_cells)) / len(core_cells)), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, sum(export_list(core_cells)) / len(core_cells)), 'red'))
        # --> If any of the parallel cells are positive, the straight var must be zero
        core_parallel = parallel[1:-1]
        if _print:
            print("{} {}".format(straight_var, '<= 1 - sum(core_parallel) / len(core_parallel)'))
        if _eval is not None:
            if _eval <= 1 - sum(export_list(core_parallel)) / len(core_parallel):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, 1 - sum(export_list(core_parallel)) / len(core_parallel)), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, 1 - sum(export_list(core_parallel)) / len(core_parallel)), 'red'))
        # --> If any core cell is an intersection, the straight var must also be zero
        core_intersections = intersections[1:-1]
        if _print:
            print("{} {}".format(straight_var, '<= 1 - sum(core_intersections) / len(core_intersections)'))
        if _eval is not None:
            if _eval <= 1 - sum(export_list(core_intersections)) / len(core_intersections):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, 1 - sum(export_list(core_intersections)) / len(core_intersections)), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, 1 - sum(export_list(core_intersections)) / len(core_intersections)), 'red'))

        # --> If any parallel cell is an intersection, the straight var must also be zero
        core_intersections = intersections[1:-1]
        if _print:
            print("{} {}".format(straight_var, '<= 1 - sum(core_intersections) / len(core_intersections)'))
        if _eval is not None:
            if _eval <= 1 - sum(export_list(core_intersections)) / len(core_intersections):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, 1 - sum(export_list(core_intersections)) / len(core_intersections)), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, 1 - sum(export_list(core_intersections)) / len(core_intersections)), 'red'))

        # --> If the start continues the straight, the straight var must be zero
        # This means the straight var may only be 1, if either the beginning cell is 0, the beginning cell is an intersection or the cell parallel to the beginning is 1
        if _print:
            print("{} {}".format(straight_var, '<= (1 - cells[0]) + intersections[0] + parallel[0]'))
        if _eval is not None:
            if _eval <= (1 - export_variable(cells[0])) + export_variable(intersections[0]) + export_variable(parallel[0]):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, (1 - export_variable(cells[0])) + export_variable(intersections[0]) + export_variable(parallel[0])), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, (1 - export_variable(cells[0])) + export_variable(intersections[0]) + export_variable(parallel[0])), 'red'))
        # --> If the end continues the straight, the straight var must be zero
        # Same constraint as for start
        if _print:
            print("{} {}".format(straight_var, '<= (1 - cells[-1]) + intersections[-1] + parallel[-1]'))
        if _eval is not None:
            if _eval <= (1 - export_variable(cells[-1])) + export_variable(intersections[-1]) + export_variable(parallel[-1]):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, (1 - export_variable(cells[-1])) + export_variable(intersections[-1]) + export_variable(parallel[-1])), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, (1 - export_variable(cells[-1])) + export_variable(intersections[-1]) + export_variable(parallel[-1])), 'red'))
        # Reverse direction must also hold: var is 0 => at least one condition not satisfied
        if _print:
            print("1 - {} {}".format(straight_var, '<= len(core_cells) - sum(core_cells) + sum(core_parallel) + sum(core_intersections) + (cells[0] - intersections[0] - parallel[0]) + (cells[-1] - intersections[-1] - parallel[-1])'))
        if _eval is not None:
            if 1 - _eval <= len(core_cells) - sum(export_list(core_cells)) + sum(export_list(core_parallel)) + sum(export_list(core_intersections)) + (value(cells[0]) - value(intersections[0]) - value(parallel[0])) + (value(cells[-1]) - value(intersections[-1]) - value(parallel[-1])):
                if _show_positive:
                    print(colored("{} <= {}".format(_eval, len(core_cells) - sum(export_list(core_cells)) + sum(export_list(core_parallel)) + sum(export_list(core_intersections)) + (value(cells[0]) - value(intersections[0]) - value(parallel[0])) + (value(cells[-1]) - value(intersections[-1]) - value(parallel[-1]))), 'green'))
            else:
                print(colored("{} <= {}".format(_eval, len(core_cells) - sum(export_list(core_cells)) + sum(export_list(core_parallel)) + sum(export_list(core_intersections)) + (value(cells[0]) - value(intersections[0]) - value(parallel[0])) + (value(cells[-1]) - value(intersections[-1]) - value(parallel[-1]))), 'red'))

        print("core: {} / {}".format(export_list(core_cells, save_name=True), len(core_cells)))
        print("core intersections: {} / {}".format(export_list(core_intersections, save_name=True), len(core_intersections)))
        print("parallel: {} / {}".format(export_list(core_parallel, save_name=True), len(core_parallel)))

        return straight_var

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

        if len(self.nodes_straights.keys()) > 0:
            num_straights = ''
            for length in self.nodes_straights.keys():
                num_straights_l = sum([int(value(v)) for v in self.nodes_straights[length]])
                if num_straights_l > 0:
                    num_straights += '(l={}, n={}) '.format(length, num_straights_l)

        intersection_str = colored('{} intersections'.format(num_intersections), 'yellow')
        _90_str = colored('{} 90 degree turns'.format(num_90s), 'blue')
        _180_str = colored('{} 180 degree turns'.format(num_180s), 'magenta')
        straights_str = colored('straights {}'.format(num_straights), 'cyan')
        print("Solution has {}, {}, {} and {}".format(intersection_str, _90_str, _180_str, straights_str))

    def get_straight_vars(self):
        positive = []
        indices = self.get_grid_indices()

        for x, y in indices:
            horizontal_straights_dict = self.node_grid_straights_horizontal[x][y]
            for length in horizontal_straights_dict.keys():
                bottom, top = horizontal_straights_dict[length]
                bottom_name = bottom
                bottom_value = value(bottom)

                if bottom_value > 0:
                    positive.append(bottom_name)

                top_name = top
                top_value = value(top)

                if top_value > 0:
                    positive.append(top_name)

            vertical_straights_dict = self.node_grid_straights_vertical[x][y]
            for length in vertical_straights_dict.keys():
                left, right = vertical_straights_dict[length]
                left_name = left
                left_value = value(left)

                if left_value > 0:
                    positive.append(left_name)

                right_name = right
                right_value = value(right)

                if right_value > 0:
                    positive.append(right_name)

        return positive


class IntersectionProblem:
    def __init__(self, intersection_indices, gap_intersection_indices=None, allow_adjacent=False, n=None, iteration_constraints=None):
        self.intersection_indices = intersection_indices
        if gap_intersection_indices is None:
            self.gap_intersection_indices = []
        else:
            self.gap_intersection_indices = gap_intersection_indices
        self.problem = LpProblem("IntersectionProblem", LpMinimize)
        self.variables = self.init_variables(self.intersection_indices, tag="std")
        self.gap_variables = self.init_variables(self.gap_intersection_indices, tag="gap")
        self.add_no_adjacency_constraints(allow_adjacent)
        if n is not None:
            self.problem += sum(self.variables + self.gap_variables) == n
        self.add_iteration_constraints(iteration_constraints)

    @staticmethod
    def init_variables(indices, tag):
        variables = [LpVariable("{}_{}".format(index, tag), cat=const.LpBinary) for index in range(len(indices))]
        return variables

    def add_no_adjacency_constraints(self, allow_adjacent):
        """
        Adjacent intersection are not allowed
        """
        # Std Intersections and Gap Intersection may never be adjacent
        for i, indices in enumerate(self.intersection_indices):
            for j, gap_indices in enumerate(self.gap_intersection_indices):
                if is_adjacent(indices, gap_indices):
                    self.problem += self.variables[i] + self.gap_variables[j] <= 1

        if not allow_adjacent:
            # If chosen, std intersection may not be adjacent among themselves
            for i, indices1 in enumerate(self.intersection_indices):
                for j, indices2 in enumerate(self.intersection_indices):
                    if is_adjacent(indices1, indices2):
                        self.problem += self.variables[i] + self.variables[j] <= 1

            # If chosen, gap intersection may not be adjacent among themselves
            for i, indices1 in enumerate(self.gap_intersection_indices):
                for j, indices2 in enumerate(self.gap_intersection_indices):
                    if is_adjacent(indices1, indices2):
                        self.problem += self.gap_variables[i] + self.gap_variables[j] <= 1

    def add_iteration_constraints(self, forced_intersections):
        """
        Add intersections that must exist
        """
        all_variables = self.variables + self.gap_variables
        for index in range(len(all_variables)):
            if index in forced_intersections:
                self.problem += all_variables[index] == 1
            else:
                self.problem += all_variables[index] == 0

    def solve(self, _print=False):
        try:
            with Capturing() as output:
                status = self.problem.solve(GUROBI(msg=0))
        except:
            print(colored('GUROBI IS NOT AVAILABLE. DEFAULTING TO CBM!', 'red'))
            status = self.problem.solve(PULP_CBC_CMD(msg=0))

        solution = [value(variable) for variable in self.variables]
        if _print:
            print("{} Solution: {}".format(LpStatus[status], solution))
        return solution, status + 1


if __name__ == '__main__':
    _quantity_constraints = [
        QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, quantity=0),
        QuantityConstraint(TrackProperties.turn_180, ConditionTypes.more_or_equals, quantity=0),
        QuantityConstraint(TrackProperties.turn_90, ConditionTypes.more_or_equals, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=2, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=3, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=4, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=5, quantity=0),
        QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.more_or_equals, length=6, quantity=0),
    ]

    # pre_solved = [[1, 2, 1], [1, 0, 1], [1, 0, 1], [2, 0, 1], [1, 0, 1], [3, 0, 1], [1, 1, 1]]
    pre_solved = [[1, 1, 1], [1, 0, 1], [1, 0, 2], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]]
    p = Problem(7, 3, imitate=pre_solved, quantity_constraints=_quantity_constraints, allow_adjacent_intersections=True, allow_gap_intersections=True)

    # p = Problem(6, 3, quantity_constraints=_quantity_constraints)
    start = time.time()
    _solution, status = p.solve(_print=False)
    end = time.time()
    print(colored("Solution {}, Time elapsed: {:.2f}s".format(LpStatus[status - 1], end - start), "green" if status > 1 else "red"))
    if status > 1:
        print_2d(_solution)
        p.get_stats()

    # horizontal = [
    #     [[0, 0], [0, 0], [0, 0]],
    #     [[0, 0], [0, 0], [0, 0]],
    #     [[0, 0], [0, 0], [0, 0]],
    #     [[0, 0], [0, 0], [0, 0]],
    #     [[0, 0], [0, 0], [0, 0]],
    #     [[0, 0], [0, 0], [0, 0]],
    #     [[0, 0], [0, 0], [0, 0]],
    # ]

    horizontal = [
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
    ]

    vertical = [
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0]],
    ]

    straight_vars = p.get_straight_vars()
    # print(straight_vars)

    p.debug_straights_constraints(2, _horizontal=horizontal)

    # for i in range(2, 7, 1):
    #     p.debug_straights_constraints(i, _horizontal=horizontal)
