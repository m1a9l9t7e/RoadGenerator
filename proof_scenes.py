import os

from manim import *

from creation_scenes import track_properties_to_colors
from fm.model import calculate_problem_dict, FeatureModel
from interpolation import get_interpolation_animation_piece_wise
from ip.ip_util import QuantityConstraint, ConditionTypes, SolutionEntries
from ip.problem import Problem
from ip.iteration import get_intersect_matrix, convert_solution_to_graph, get_custom_solution, get_solution_from_config
from presentation_scenes import get_legend
from util import Grid, GridShowCase, draw_graph, get_square, get_text, get_arrow, remove_graph, generate_track_points, TrackProperties, extract_graph_tours, print_2d
from graph import Graph, Node
from anim_sequence import AnimationObject, AnimationSequenceScene, make_concurrent


class ProofVisualization:
    def __init__(self, show_text='names'):
        if show_text not in ['names', 'values']:
            raise ValueError('Unknown value for show_text: {}'.format(show_text))

        self.scale = 1
        self.solution = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.width, self.height = [value+1 for value in np.shape(self.solution)]
        self.ip_width, self.ip_height = (self.width - 1, self.height - 1)
        self.show_text = show_text
        self.n = (self.width * self.height - 4) / 2 + 1
        self.num_cells = self.ip_width * self.ip_height
        self.num_nodes = self.width * self.height
        self.helper = GridShowCase(self.num_cells, [self.scale, self.scale], spacing=[0, 0], space_ratio=[self.ip_width, self.ip_height])
        self.animation_sequence = []

        self.node_row_helper = GridShowCase(self.num_nodes, [0.2 * self.scale, 0.2 * self.scale],
                                            spacing=[0.2 * self.scale, 0.2 * self.scale], space_ratio=[self.num_nodes, 1],
                                            shift=[-self.scale * 2, self.height * self.scale + 0.3])
        self.node_counter = 0
        self.removed_nodes = []
        self.removed_nodes_by_coords = []
        self.moved_nodes = []

        self.initial_square = None
        self.other_squares = []

        # Descriptors
        self.negative_cell_desc = (BLACK, DARK_GREY, 'Negative Cell')
        self.positive_cell_desc = (GREEN_E, DARK_GREY, 'Positive Cell')
        self.intersection_cell_desc = (YELLOW_E, DARK_GREY, 'Intersection')
        self.root_cell_desc = (DARK_BROWN, DARK_GREY, 'Root Cell')

        # Pre-declarations
        self.captions = []
        self.legend = []
        self.graph = None

        # animation times
        self.t_draw_graph = 0.75
        self.t_add_cell = 1
        self.t_move_nodes = 1
        self.t_add_brace = 1
        self.t_move_cells = 2

    def get_animation_sequence(self):
        pc1, sc1, _ = self.positive_cell_desc
        mini_square_size = 0.2

        self.add_graph()
        self.add_squares()
        # self.add_legend()
        self.add_single_square(0, 0)
        self.add_single_square(0, 1)
        self.add_single_square(0, 2)
        self.add_single_square(1, 1)
        self.add_single_square(2, 1)
        self.add_single_square(2, 0)
        self.add_single_square(2, 2)

        # Make Curly Brace under inital four nodes
        eq_group = VGroup(*self.moved_nodes[:4])
        braces = Brace(eq_group, DOWN, stroke_width=0.1)
        # eq_text = braces.get_text("4 nodes")
        eq_text = braces.get_text("4 nodes").scale(0.75*self.scale).shift([0, 0.2*self.scale, 0])
        below_braces_coords = (eq_text.get_x(), eq_text.get_y())

        self.animation_sequence.append(AnimationObject('play', content=GrowFromCenter(braces), duration=self.t_add_brace))
        # self.animation_sequence.append(AnimationObject('play', content=Write(eq_text), duration=1))

        # Show that initial cell covers these
        below_first_braces_coords = below_braces_coords
        new_initial_square = get_square(below_first_braces_coords, mini_square_size, pc1, sc1, border_width=0)
        self.animation_sequence.append(AnimationObject('play', content=ReplacementTransform(self.initial_square, new_initial_square), duration=self.t_move_cells, z_index=10))

        # Make Curly Brace under all other nodes
        eq_group = VGroup(*self.moved_nodes[4:])
        braces = Brace(eq_group, DOWN, stroke_width=0.1)
        # eq_text = braces.get_text("$mn - 4$ nodes")
        eq_text = braces.get_text("$mn - 4$ nodes").scale(0.75*self.scale).shift([0, 0.2*self.scale, 0])
        below_braces_coords = (eq_text.get_x(), eq_text.get_y())
        self.animation_sequence.append(AnimationObject('play', content=GrowFromCenter(braces), duration=self.t_add_brace, z_index=10))
        # self.animation_sequence.append(AnimationObject('play', content=Write(eq_text), duration=1)) # Text under brace

        # Show which squares are responsible
        square_row_helper = GridShowCase(self.num_nodes, [mini_square_size * self.scale, mini_square_size * self.scale],
                                            spacing=[mini_square_size * self.scale, mini_square_size * self.scale], space_ratio=[len(self.other_squares), 1],
                                            shift=below_braces_coords)
        _, square_row_size = square_row_helper.get_global_camera_settings()
        square_row_helper.shift = (square_row_helper.shift[0] - square_row_size[0] / 3.75, square_row_helper.shift[1])

        mini_squares = []
        move_other_squares_animations = []
        for index, square in enumerate(self.other_squares):
            coords = square_row_helper.get_element_coords(index)
            new_square = get_square(coords, mini_square_size, pc1, sc1, border_width=0)
            # move_other_squares_animations.append(Transform(square, new_square))
            move_other_squares_animations.append(ReplacementTransform(square, new_square))
            mini_squares.append(new_square)

        self.animation_sequence.append(AnimationObject('play', content=move_other_squares_animations, duration=self.t_move_cells, z_index=10))

        # init square and plus
        x, y = below_first_braces_coords
        one = get_text('$1$', (x, y))
        plus = get_text('$+$', (x+0.8, y))
        self.animation_sequence.append(AnimationObject('play', content=ReplacementTransform(new_initial_square, one), duration=2))
        self.animation_sequence.append(AnimationObject('play', content=Create(plus), duration=1))

        # k equation
        mini_square_group = VGroup(*mini_squares)
        moved_nodes_group = VGroup(*self.moved_nodes[4:])

        x, y = mini_square_group.get_center()[:2]
        x_shift = 0
        x -= 0.5
        right_equation_shift = 1.2
        k = get_text('$k$', (x - 0.5, y))
        equals = get_text('$=$', (x, y - 0.05))
        counter = get_text('$mn - 4$', (x + right_equation_shift, y))
        hline = get_text('\\rule{1.1cm}{0.01cm}', (x + right_equation_shift, y - 0.25))
        denominator = get_text('$2$', (x + right_equation_shift, y - 0.5))
        self.animation_sequence.append(AnimationObject('play', content=ReplacementTransform(mini_square_group, k), duration=2))
        self.animation_sequence.append(AnimationObject('play', content=Create(equals), duration=1))
        self.animation_sequence.append(AnimationObject('play', content=ReplacementTransform(moved_nodes_group, counter), duration=2))
        self.animation_sequence.append(AnimationObject('play', content=Create(hline), duration=1))
        self.animation_sequence.append(AnimationObject('play', content=Create(denominator), duration=1))

        # N
        x, y = below_first_braces_coords
        n_equals = get_text('$N =$', (x-1, y))
        self.animation_sequence.append(AnimationObject('play', content=Create(n_equals), duration=1))
        return self.animation_sequence

    def add_pause(self, duration):
        self.animation_sequence += [AnimationObject('wait', content=[], wait_after=duration)]

    def add_squares(self):
        # unpack primary and secondary colors
        pc1, sc1, _ = self.positive_cell_desc
        pc2, sc2, _ = self.negative_cell_desc
        pc3, sc3, _ = self.intersection_cell_desc
        pc4, sc4, _ = self.root_cell_desc

        self.squares = [[None for y in range(self.ip_height)] for x in range(self.ip_width)]
        self.squares_flat = []

        self.captions = []
        counter = 0
        for y in range(self.ip_height):
            for x in range(self.ip_width):
                coords = self.helper.get_element_coords(counter)
                counter += 1
                if self.solution[x][y] == SolutionEntries.positive:
                    primary_color, secondary_color, _ = self.positive_cell_desc
                    square = get_square(coords, self.scale, primary_color, secondary_color, border_width=2 * self.scale)
                    if x == 0 and y == 0 and self.show_root:
                        primary_color, secondary_color, _ = self.root_cell_desc
                        square = get_square(coords, self.scale, primary_color, secondary_color, border_width=2 * self.scale)
                elif self.solution[x][y] == SolutionEntries.positive_and_intersection:
                    primary_color, secondary_color, _ = self.intersection_cell_desc
                    square = get_square(coords, self.scale, primary_color, secondary_color, border_width=2 * self.scale)
                elif self.solution[x][y] == SolutionEntries.negative_and_intersection:
                    primary_color, secondary_color, _ = self.root_cell_desc
                    square = get_square(coords, self.scale, primary_color, secondary_color, border_width=2 * self.scale)
                elif self.solution[x][y] == SolutionEntries.negative:
                    primary_color, secondary_color, _ = self.negative_cell_desc
                    square = get_square(coords, self.scale, primary_color, secondary_color, border_width=2 * self.scale)

                self.squares[x][y] = square
                self.squares_flat.append(square)

                if self.show_text == 'values' and self.solution[x][y] in [SolutionEntries.positive, SolutionEntries.positive_and_intersection]:
                    node_grid_values = self.problem_dict['node_grid_values']
                    self.captions.append(get_text('{}'.format(node_grid_values[x][y]), coords))
                elif self.show_text == 'names':
                    self.captions.append(get_text(r'$c_{' + str(x+1) + ',' + str(y+1) + '}$', coords))

        self.animation_sequence += [
            # AnimationObject('add', content=self.squares_flat, z_index=-5),
            # AnimationObject('play', content=[Create(text) for text in self.captions], duration=1, z_index=5)
            AnimationObject('add', content=self.captions, z_index=20)
        ]

    def add_single_square(self, x, y):
        pc1, sc1, _ = self.positive_cell_desc
        pc2, sc2, _ = self.negative_cell_desc
        pc3, sc3, _ = self.intersection_cell_desc
        pc4, sc4, _ = self.root_cell_desc

        coords = self.helper.get_element_coords2d(x, y)
        primary_color, secondary_color, _ = self.positive_cell_desc
        square = get_square(coords, self.scale, primary_color, secondary_color, border_width=0)

        if self.node_counter == 0:
            self.initial_square = square
        else:
            self.other_squares.append(square)

        covered_nodes = self.graph.get_covered_nodes((x, y))
        newly_covered_nodes = []

        for node in covered_nodes:
            if node.get_coords() in self.removed_nodes_by_coords:
                continue
            else:
                newly_covered_nodes.append(node)
                self.removed_nodes_by_coords.append(node.get_coords())

        node_drawables = [node.drawable for node in newly_covered_nodes]
        node_animations = []

        for node_drawable in node_drawables:
            _x, _y = self.node_row_helper.get_element_coords(self.node_counter)
            new_node = Node(_x, _y)
            node_animations.append(Transform(node_drawable, new_node.drawable))
            self.moved_nodes.append(new_node.drawable)
            self.node_counter += 1

        # square_animation = Transform(self.squares[x][y], square)
        square_animation = FadeIn(square)

        self.animation_sequence += [
            AnimationObject('play', content=square_animation, duration=self.t_add_cell, z_index=-5),
            AnimationObject('play', content=node_animations, duration=self.t_move_nodes)
        ]

    def remove_squares(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.squares),
        ]

    def remove_captions(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.captions),
        ]

    def add_graph(self):
        self.graph = Graph(self.width, self.height, shift=[-self.scale / 2, -self.scale / 2])
        self.animation_sequence += draw_graph(self.graph, z_index=5, duration=self.t_draw_graph)
        return

    def remove_graph_edges(self):
        return [AnimationObject(type='remove', content=[edge.drawable for edge in self.graph.edges])]

    def remove_graph_nodes(self):
        return [AnimationObject(type='remove', content=[node.drawable for node in self.graph.nodes])]

    def add_legend(self):
        legend_entries = [
            self.negative_cell_desc,
            self.positive_cell_desc
        ]
        legend_entries.insert(0, self.intersection_cell_desc)
        legend_entries.insert(0, self.root_cell_desc)

        _, camera_size, _ = self.get_camera_settings()
        self.legend = get_legend(legend_entries, shift=[camera_size[0] + self.scale / 4, self.scale / 2], scale=self.scale * 0.5)
        self.animation_sequence += [
            AnimationObject('add', content=self.legend, bring_to_front=True)
        ]

    def remove_legend(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.legend)
        ]

    def get_camera_settings(self):
        camera_position, camera_size = self.helper.get_global_camera_settings()
        shift = [-self.scale / 2, -self.scale / 2]
        return camera_position, camera_size, shift


class IPExtra(AnimationSequenceScene):
    def construct(self):
        viz = ProofVisualization()
        camera_position, camera_size, shift = viz.get_camera_settings()
        shift = [shift[0], shift[1] + 0.9]
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.9, shift=shift)
        self.play_animations(viz.get_animation_sequence())
        self.wait(5)


if __name__ == '__main__':
    scene = IPExtra()
    scene.construct()
