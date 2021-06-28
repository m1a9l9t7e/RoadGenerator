from anim_sequence import AnimationObject
from graph import Graph, GraphSearcher
from util import draw_graph, make_unitary, GridShowCase


class GraphModel:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.leaf_sequences = []
        self.hash_map = dict()
        self.iterate()

    def iterate(self):
        queue = [JoinSequence(self.width, self.height)]
        hash_map = dict()
        while len(queue) > 0:
            sequence = queue.pop(0)
            continuations = sequence.get_possible_continuations()
            if len(continuations) == 0:
                key = str(sequence.generate_graph())
                if key not in hash_map:
                    self.leaf_sequences.append(sequence)
                    hash_map[key] = True
            else:
                queue += continuations

        print("Possible tours: {}".format(len(self.leaf_sequences)))

    def get_graphs(self):
        graph_list = []
        for sequence in self.leaf_sequences:
            graph = sequence.generate_graph()
            graph_list.append(graph)

        return graph_list

    def get_animations(self, scale, ratio=[16, 9], spacing=[1, 1]):
        helper = GridShowCase(num_elements=len(self.leaf_sequences),
                              element_dimensions=(scale * self.width, scale * self.height),
                              spacing=spacing, space_ratio=ratio)
        animations_list = []
        graph_list = []
        for index, sequence in enumerate(self.leaf_sequences):
            shift = helper.get_element_coords(index)
            animations, graph = sequence.get_animations(scale=scale, shift=shift)
            animations_list.append(animations)
            graph_list.append(graph)

        return animations_list, graph_list, helper


class JoinSequence:
    def __init__(self, width, height, sequence=None):
        self.width = width
        self.height = height
        self.sequence = [] if sequence is None else sequence

    def __str__(self):
        _str = "Join Sequence: "
        for idx, (index, operation) in enumerate(self.sequence):
            _str += "{}.({},{});".format(idx, operation, index)
        return _str

    def get_possible_continuations(self):
        continuations = []
        graph = self.generate_graph()
        searcher = GraphSearcher(graph)
        joints = searcher.walk_graph()
        for i in range(len(joints)):
            continuations.append(self.get_new_sequence((i, 'intersect')))
            continuations.append(self.get_new_sequence((i, 'merge')))

        return continuations

    def get_new_sequence(self, continuation):
        continued_sequence = self.sequence + [continuation]
        return JoinSequence(self.width, self.height, continued_sequence)

    def generate_graph(self):
        graph = Graph(width=self.width, height=self.height)
        graph.remove_all_but_unitary()
        graph.init_cycles()
        for (index, operation) in self.sequence:
            searcher = GraphSearcher(graph)
            joints = searcher.walk_graph()
            joint = joints[index]
            if operation == 'intersect':
                joint.intersect()
            elif operation == 'merge':
                joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))

        return graph

    def get_animations(self, scale, shift):
        animations = []
        graph = Graph(width=self.width, height=self.height, scale=scale, shift=shift)
        animations += draw_graph(graph)
        animations += make_unitary(graph)
        graph.init_cycles()
        for (index, operation) in self.sequence:
            searcher = GraphSearcher(graph)
            joints = searcher.walk_graph()
            animations.append(AnimationObject(type='add', content=[joint.drawable for joint in joints], wait_after=1))
            joint = joints[index]
            animations.append(AnimationObject(type='remove', content=joint.drawable))
            if operation == 'intersect':
                animations += joint.intersect()
            elif operation == 'merge':
                animations += joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))

            animations.append(AnimationObject(type='remove', content=[joint.drawable for joint in joints], wait_after=1))

        return animations, graph
