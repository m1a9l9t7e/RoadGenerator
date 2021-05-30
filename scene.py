from manimlib import *
# from manim import *
from graph import Graph, GraphSearcher


class CircuitCreation(Scene):
    def construct(self):
        self.create_graph(4, 4)
        self.two_factorization()
        # self.search_graph()
        self.custom_joins()
        self.wait(2)

    def create_graph(self, width=4, height=4):
        # Get Graph
        self.graph = Graph(width, height)
        nodes = self.graph.nodes
        edges = self.graph.edges
        # self.play(
        #     # Set the size with the width of a object
        #     self.camera_frame.set_width, width * 1.7,
        #     # Move the camera to the object
        #     self.camera_frame.move_to, (width/2.5, height/2.5, 0)
        # )

        # Draw Nodes
        for node in nodes:
            circle = node.drawable
            self.add(circle)

        # Draw Edges
        animations = []
        for edge in edges:
            line = edge.drawable
            self.bring_to_back(line)
            animations.append(ShowCreation(line))

        self.play(*animations, run_time=1)

    def make_unitary(self):
        drawables = self.graph.remove_all_but_unitary()
        animations = [FadeOut(drawable) for drawable in drawables]
        self.play(*animations, run_time=0.5)

    def two_factorization(self):
        drawables_list = self.graph.two_factorization()
        for drawables in drawables_list:
            for drawable in drawables:
                self.play(FadeOut(drawable), run_time=0.1)

    def search_graph(self):
        searcher = GraphSearcher(self.graph)
        joints = searcher.walk_graph()
        for joint in joints:
            self.add(joint.drawable)

        self.wait(duration=1)

        for idx, joint in enumerate(joints):
            self.remove(joint.drawable)
            if idx >= self.graph.cycles - 1:
                break
            # animations = joint.intersect()
            animations = joint.merge()
            self.play(*animations, run_time=3)

    def custom_joins(self):
        """
        Custom joins specifically designed to resemble
        example circuit from ruleset starting from 4x4 grid
        """
        searcher = GraphSearcher(self.graph)
        joints = searcher.walk_graph()
        for joint in joints:
            self.add(joint.drawable)

        self.wait(duration=1)

        indices = [0, 3, 2]
        operations = ['intersect', 'merge', 'intersect']

        for i, idx in enumerate(indices):
            joint = joints[idx]
            self.remove(joint.drawable)
            operation = operations[i]
            if operation == 'intersect':
                animations = joint.intersect()
            elif operation == 'merge':
                animations = joint.merge()
            else:
                raise ValueError('operation "{}" is undefined!'.format(operation))
            self.play(*animations, run_time=3)

        for joint in joints:
            self.remove(joint.drawable)
