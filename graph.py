from manim import *
from anim_sequence import AnimationObject
from util import Converter, transform_coords
import copy

xvec = [1, 0, -1, 0]
yvec = [0, 1, 0, -1]


class Node:
    def __init__(self, x, y, real_coords=None):
        self.x = x
        self.y = y
        if real_coords is None:
            self.real_coords = (x, y, 0)
        else:
            self.real_coords = real_coords
        self.edges = []
        self.adjacent_nodes = []
        self.drawable = self._create_drawable()
        self.cycle_id = None

    def __str__(self):
        adjacent = "adjacent:\n"
        for idx, adjacent_node in enumerate(self.adjacent_nodes):
            adjacent += "\t{}. ({},{})".format(idx + 1, adjacent_node.x, adjacent_node.y)
        # return "Node ({},{})\n{}".format(self.x, self.y, adjacent)
        return "Node ({},{})".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def get_coords(self):
        return self.x, self.y

    def get_real_coords(self):
        return self.real_coords

    def get_degree(self):
        return len(self.adjacent_nodes)

    def is_adjacent_to(self, node):
        for edge in self.edges:
            if node == edge.node1 or node == edge.node2:
                return True
        return False

    def add_edge(self, edge):
        if edge.node1 == self:
            self.adjacent_nodes.append(edge.node2)
        elif edge.node2 == self:
            self.adjacent_nodes.append(edge.node1)
        else:
            raise Exception("Edge not adjacent to this Node!")
        self.edges.append(edge)

    def get_edge_to(self, adjacent_node):
        for edge in self.edges:
            if adjacent_node == edge.node1 or adjacent_node == edge.node2:
                return edge

    def remove_adjacent_node(self, node):
        edge = self.get_edge_to(node)
        return edge.remove()

    def _create_drawable(self):
        circle = Dot(point=self.get_real_coords(), radius=0.1)
        circle.set_fill(BLUE, opacity=1)
        circle.set_stroke(BLUE_E, width=4)
        return circle


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.drawable = self._create_drawable()
        node1.add_edge(self)
        node2.add_edge(self)

    def __str__(self):
        return "Edge {} - {}".format(self.node1.get_coords(), self.node2.get_coords())

    def __eq__(self, other):
        return self.node1 == other.node1 and self.node2 == other.node2

    def swap_nodes(self):
        swap = self.node1
        self.node1 = self.node2
        self.node2 = swap

    def _create_drawable(self):
        line = Line(self.node1.get_real_coords(), self.node2.get_real_coords())
        return line

    def remove(self):
        self.node1.adjacent_nodes.remove(self.node2)
        self.node2.adjacent_nodes.remove(self.node1)
        self.node1.edges.remove(self)
        self.node2.edges.remove(self)
        return self.drawable


class Graph:
    def __init__(self, width, height, scale=1, shift=[0, 0]):
        self.grid = [[Node(-1, -1)] * height for i in range(width)]
        self.nodes = []
        self.edges = []
        for x in range(width):
            for y in range(height):
                node = Node(x, y, real_coords=transform_coords([x, y], shift, scale))
                self.grid[x][y] = node
                self.nodes.append(node)

        for x in range(width):
            for y in range(height):
                node = self.grid[x][y]
                right = (x + 1, y)
                above = (x, y + 1)
                for coords in [right, above]:
                    if self.get_element_safe(coords) is not None:
                        neighbor_node = self.get_element_safe(coords)
                        edge = Edge(node, neighbor_node)
                        self.edges.append(edge)

        self.cycles = int(width / 2) * int(height / 2)  # upper bound

    def __eq__(self, other):
        self_map = dict()
        other_map = dict()
        for edge in other.edges:
            other_map[str(edge)] = True

        # check if all keys of self are in other
        for edge in self.edges:
            key = str(edge)
            self_map[key] = True
            if key not in other_map:
                # print("({}) FROM SELF NOT IN OTHER".format(key))
                return False

        # check if all keys of other are in self
        for edge in other.edges:
            key = str(edge)
            if key not in self_map:
                # print("({}) FROM OTHER NOT IN SELF".format(key))
                return False

        return True

    def get_element_safe(self, coords):
        x = coords[0]
        y = coords[1]
        if x >= len(self.grid) or y >= len(self.grid[x]):
            return None
        else:
            return self.grid[x][y]

    def remove_all_but_unitary(self):
        drawables = []
        removed_edges = []
        for edge in self.edges:
            bad_horizontal = edge.node1.x % 2 != 0 and edge.node2.x % 2 == 0
            bad_vertical = edge.node1.y % 2 != 0 and edge.node2.y % 2 == 0
            if bad_horizontal or bad_vertical:
                removed_edges.append(edge)
                drawable = edge.remove()
                drawables.append(drawable)

        for edge in removed_edges:
            self.edges.remove(edge)

        return drawables

    def get_max_degree(self):
        max_degree = 0
        argmax = None
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                node = self.grid[x][y]
                if node.get_degree() > max_degree:
                    max_degree = node.get_degree()
                    argmax = node

        return max_degree, argmax

    def bfs(self, node, blocked_node=None):
        """
        Perform grid search
        :param node: Node where search starts.
        :param blocked_node: Node that blocks search.
        :return: Grid where reachable positions are marked with 1, unreachable 0 and blocked -1
        """
        queue = [node]
        grid = [[0] * len(self.grid[0]) for i in range(len(self.grid))]
        grid[node.x][node.y] = -1
        if blocked_node is not None:
            x, y, = blocked_node.get_coords()
            grid[x][y] = -1
        while len(queue) > 0:
            node = queue.pop()
            for adjacent_node in node.adjacent_nodes:
                if grid[adjacent_node.x][adjacent_node.y] == 0:
                    grid[adjacent_node.x][adjacent_node.y] = 1
                    queue.append(adjacent_node)
                elif grid[adjacent_node.x][adjacent_node.y] == -1:
                    grid[adjacent_node.x][adjacent_node.y] = 1

        return grid

    def two_factorization(self):
        drawables_list = []
        max_degree, start_node = self.get_max_degree()
        while max_degree > 2:
            drawables = self.find_cycle(start_node)
            drawables_list.append(drawables)
            max_degree, start_node = self.get_max_degree()

        drawables = self.remove_non_cycles()
        drawables_list.append(drawables)
        return drawables_list

    def find_cycle(self, start_node):
        drawables = []
        current_node = start_node
        previous_node = Node(-1, -1)

        while current_node != start_node or len(drawables) == 0:
            # Eliminate nodes that won't lead back to start node
            for adjacent_node in current_node.adjacent_nodes:
                if adjacent_node == previous_node:
                    continue
                bfs_grid = self.bfs(adjacent_node, current_node)
                x, y, = start_node.get_coords()
                if bfs_grid[x][y] < 1:
                    drawable = current_node.remove_adjacent_node(adjacent_node)
                    drawables.append(drawable)

            # Leave only the two nodes with the lowest degree
            while current_node.get_degree() != 2:
                max_degree = 0
                argmax = None
                for adjacent_node in current_node.adjacent_nodes:
                    if adjacent_node == previous_node:
                        continue
                    if adjacent_node.get_degree() > max_degree:
                        max_degree = adjacent_node.get_degree()
                        argmax = adjacent_node
                if argmax is not None:
                    drawable = current_node.remove_adjacent_node(argmax)
                    drawables.append(drawable)
                else:
                    break

            # Move on to next node
            if current_node.get_degree() == 2:
                node1, node2 = current_node.adjacent_nodes
                if node1 == previous_node:
                    previous_node = current_node
                    current_node = node2
                elif node2 == previous_node or previous_node == Node(-1, -1):
                    previous_node = current_node
                    current_node = node1
                else:
                    raise RuntimeError("Something went wrong")
            else:
                break

        return drawables

    def remove_non_cycles(self):
        drawables = []
        while True:
            change = False
            for node in self.nodes:
                if node.get_degree() == 1:
                    drawable = node.remove_adjacent_node(node.adjacent_nodes[0])
                    drawables.append(drawable)
                    change = True
            if not change:
                break
        return drawables

    def join_dots(self):
        pass

    def init_cycles(self, print_debug=False):
        counter = 0
        cycle_map = [[None] * len(self.grid[0]) for _ in range(len(self.grid))]
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                node = self.grid[x][y]
                if node.cycle_id is None:
                    bfs_map = self.bfs(node)
                    for _x in range(len(bfs_map)):
                        for _y in range(len(bfs_map[_x])):
                            if bfs_map[_x][_y] == 1:
                                cycle_map[_x][_y] = counter
                                self.grid[_x][_y].cycle_id = counter
                    counter += 1
        if print_debug:
            print_2d(cycle_map)
        self.cycles = counter

    def merge_cycles(self, cycle_id1, cycle_id2):
        merge_id = cycle_id1
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                node = self.grid[x][y]
                if node.cycle_id == cycle_id1 or node.cycle_id == cycle_id2:
                    node.cycle_id = merge_id


class GraphSearcher:
    def __init__(self, graph):
        self.graph = graph
        self.grid = graph.grid

    def walk_graph(self):
        joints = []
        for x in range(len(self.grid) - 1):
            for y in range(len(self.grid[x]) - 1):
                coords = (x, y)
                maybe_joint = self.evaluate_position(coords)
                if maybe_joint:
                    joints.append(maybe_joint)

        return joints

    def evaluate_position(self, coords):
        x, y = coords
        bottom_left = self.grid[x][y]
        bottom_right = self.grid[x + 1][y]
        top_left = self.grid[x][y + 1]
        top_right = self.grid[x + 1][y + 1]
        # Clockwise starting at bottom left
        corners = [bottom_left, top_left, top_right, bottom_right]
        cycle_ids = [node.cycle_id for node in corners]
        # Clockwise starting at bottom
        adjacency = [bottom_left.is_adjacent_to(bottom_right), bottom_left.is_adjacent_to(top_left),
                     top_left.is_adjacent_to(top_right), bottom_right.is_adjacent_to(top_right)]

        # Check for Horizontal Joint
        parallel_horizontally = adjacency[0] and adjacency[2] and not (adjacency[1] or adjacency[3])
        distinct_cycles = cycle_ids[0] == cycle_ids[3] and cycle_ids[1] == cycle_ids[2] and cycle_ids[0] != cycle_ids[1]
        if parallel_horizontally and distinct_cycles:
            return HorizontalJoint(corners, cycle_ids, adjacency, self.graph)

        # Check for Vertical Joint
        parallel_vertically = adjacency[1] and adjacency[3] and not (adjacency[0] or adjacency[2])
        distinct_cycles = cycle_ids[0] == cycle_ids[1] and cycle_ids[2] == cycle_ids[3] and cycle_ids[1] != cycle_ids[2]
        if parallel_vertically and distinct_cycles:
            return VerticalJoint(corners, cycle_ids, adjacency, self.graph)

        return None


class Joint:
    def __init__(self, corners, cycle_ids, adjacency, graph):
        self.corners = corners
        self.cycle_ids = cycle_ids
        self.adjacency = adjacency
        self.graph = graph
        self.drawable = self._create_drawable()

    def _create_drawable(self):
        x1, y1, z1 = self.corners[0].get_real_coords()
        x2, y2, z2 = self.corners[2].get_real_coords()
        circle = Dot(point=((x1 + x2) / 2, (y1 + y2) / 2, z1), radius=0.1)
        circle.set_fill(RED, opacity=1)
        circle.set_stroke(RED_E, width=4)
        return circle

    def _update_graph(self, deleted_edges, new_edges):
        unique_cycle_ids = set(self.cycle_ids)
        if len(unique_cycle_ids) != 2:
            raise ValueError("Trying to merge {} cycles instead of 2!".format(len(unique_cycle_ids)))

        self.graph.merge_cycles(*set(self.cycle_ids))
        for edge in deleted_edges:
            self.graph.edges.remove(edge)
        self.graph.edges += new_edges

    @staticmethod
    def _create_animation_sequence(replacements):
        new_drawables = []
        animations = []
        for replacement in replacements:
            old_drawable, new_drawable = replacement
            new_drawables.append(new_drawable)
            animations.append(ReplacementTransform(old_drawable, new_drawable))

        animation_sequence = [
            AnimationObject(type='play', content=animations, duration=1),
            AnimationObject(type='add', content=new_drawables, bring_to_back=True)
        ]

        return animation_sequence

    def intersect(self):
        pass

    def merge(self):
        pass


class VerticalJoint(Joint):
    def intersect(self):
        edge1_old = self.corners[0].get_edge_to(self.corners[1])
        edge1_old_drawable = self.corners[0].remove_adjacent_node(self.corners[1])
        edge2_old = self.corners[3].get_edge_to(self.corners[2])
        edge2_old_drawable = self.corners[3].remove_adjacent_node(self.corners[2])
        edge1_new = Edge(self.corners[3], self.corners[1])
        edge2_new = Edge(self.corners[0], self.corners[2])
        self._update_graph([edge1_old, edge2_old], [edge1_new, edge2_new])
        return self._create_animation_sequence([[edge1_old_drawable, edge1_new.drawable], [edge2_old_drawable, edge2_new.drawable]])

    def merge(self):
        edge1_old = self.corners[0].get_edge_to(self.corners[1])
        edge1_old_drawable = self.corners[0].remove_adjacent_node(self.corners[1])
        edge2_old = self.corners[3].get_edge_to(self.corners[2])
        edge2_old_drawable = self.corners[3].remove_adjacent_node(self.corners[2])
        edge1_new = Edge(self.corners[0], self.corners[3])
        edge2_new = Edge(self.corners[1], self.corners[2])
        self._update_graph([edge1_old, edge2_old], [edge1_new, edge2_new])
        return self._create_animation_sequence([[edge1_old_drawable, edge1_new.drawable], [edge2_old_drawable, edge2_new.drawable]])


class HorizontalJoint(Joint):
    def intersect(self):
        edge1_old = self.corners[0].get_edge_to(self.corners[3])
        edge1_old_drawable = self.corners[0].remove_adjacent_node(self.corners[3])
        edge2_old = self.corners[1].get_edge_to(self.corners[2])
        edge2_old_drawable = self.corners[1].remove_adjacent_node(self.corners[2])
        edge1_new = Edge(self.corners[0], self.corners[2])
        edge2_new = Edge(self.corners[1], self.corners[3])
        self._update_graph([edge1_old, edge2_old], [edge1_new, edge2_new])
        return self._create_animation_sequence([[edge1_old_drawable, edge1_new.drawable], [edge2_old_drawable, edge2_new.drawable]])

    def merge(self):
        edge1_old = self.corners[0].get_edge_to(self.corners[3])
        edge1_old_drawable = self.corners[0].remove_adjacent_node(self.corners[3])
        edge2_old = self.corners[1].get_edge_to(self.corners[2])
        edge2_old_drawable = self.corners[1].remove_adjacent_node(self.corners[2])
        edge1_new = Edge(self.corners[0], self.corners[1])
        edge2_new = Edge(self.corners[3], self.corners[2])
        self._update_graph([edge1_old, edge2_old], [edge1_new, edge2_new])
        return self._create_animation_sequence([[edge1_old_drawable, edge1_new.drawable], [edge2_old_drawable, edge2_new.drawable]])


def print_2d(array):
    print_str = ""
    for x in range(len(array)):
        print_str += " ".join([str(value) for value in array[x]]) + "\n"

    print(print_str)


class GraphModel:
    def __init__(self, base_graph):
        self.base_graph = base_graph

    def iterate_all_possible_tours(self):
        g1 = copy.deepcopy(self.base_graph)
        g2 = copy.deepcopy(self.base_graph)
        g1_search = GraphSearcher(g1)
        g2_search = GraphSearcher(g2)
        g1_joints = g1_search.walk_graph()
        g2_joints = g2_search.walk_graph()
        g1_joints[0].merge()
        g2_joints[0].intersect()

        print("g1 == g1: {}".format(g1==g1))
        print("g1 == g2: {}".format(g1==g2))
        return g1, g2

        # queue = []
        # queue.append(self.base_graph)
        # pass

    def get_next(self, graph):
        searcher = GraphSearcher(g)
        joints = searcher.walk_graph()
        if len(joints) == 0:
            return []

        # animation_sequence.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))
        # joint = joints[0]
        # animation_sequence.append(AnimationObject(type='remove', content=joint.drawable))
        # if random.choice([True, False]):
        #     animation_sequence += joint.merge()
        # else:
        #     animation_sequence += joint.intersect()
        # animation_sequence.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints]))


if __name__ == '__main__':
    # init graph
    g = Graph(4, 4)
    g.remove_all_but_unitary()
    g.init_cycles()

    gm = GraphModel(g)
    gm.iterate_all_possible_tours()

    # find joints and merge until single cycle
    # searcher = GraphSearcher(g)
    # while True:
    #     joints = searcher.walk_graph()
    #     if len(joints) == 0:
    #         break
    #     # join first joint
    #     joint = joints[0]
    #     joint.merge()
    #
    # # Convert to geometrics
    # converter = Converter(g, 2, 1)
    # converter.extract_tour()
