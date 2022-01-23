import os
from pprint import pprint

import numpy as np
from manim import *

from creation_scenes import track_properties_to_colors
from fm.model import calculate_problem_dict, FeatureModel
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous
from ip.ip_util import QuantityConstraint, ConditionTypes, SolutionEntries
from ip.problem import Problem
from ip.iteration import get_intersect_matrix, convert_solution_to_graph, get_custom_solution, get_solution_from_config
from util import Grid, GridShowCase, draw_graph, get_square, get_text, get_arrow, remove_graph, generate_track_points, TrackProperties, extract_graph_tours, print_2d
from graph import Graph
from anim_sequence import AnimationObject, AnimationSequenceScene, make_concurrent


class Basics(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1.3
        track_width = 0.4

        self.play(
            self.camera.frame.animate.set_width(width * square_size * 2.1),
            run_time=0.1
        )
        self.play(
            self.camera.frame.animate.move_to((square_size * width / 2.5, square_size * height / 2.5, 0)),
            run_time=0.1
        )

        graph = Graph(width, height, scale=square_size)
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)

        rm_idx = [0, 2, 4, 9, 11, 13, 14, 16, 18]
        rm_edges = []
        for idx, edge in enumerate(graph.edges):
            if idx in rm_idx:
                rm_edges.append(edge)

        for edge in rm_edges:
            graph.edges.remove(edge)
            edge.remove()

        # self.add(graph.)
        graph_creation = draw_graph(graph)
        self.play_animations(graph_creation)
        self.wait(5)


class IP(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1
        # track_width = 0.4
        show_graph = True
        show_intersections = False

        ip_width, ip_height = (width - 1, height - 1)
        n = np.ceil(ip_width / 2) * ip_height + np.floor(ip_width / 2)
        num_elements = ip_width * ip_height
        helper = GridShowCase(num_elements, [square_size, square_size], spacing=[0, 0], space_ratio=[1, 1])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1, shift=[-square_size/2, -square_size/2])
        # ggmst_problem = GGMSTProblem(width - 1, height - 1, raster=False)
        # solution, status = ggmst_problem.solve(_print=False)

        # p = GGMSTProblem(5, 5, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [0, 2], [1, 2], [2, 2], [2, 3], [2, 4], [1, 4], [0, 4],[0, 3]])
        # p = GGMSTProblem(5, 5, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [0, 2], [1, 2], [2, 2], [2, 3], [2, 4], [1, 4], [0, 4], [0, 3]])
        # p = GGMSTProblem(5, 5, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 4], [2, 0], [2, 2], [2, 3], [2, 4], [3, 0], [3, 3], [4, 0], [4, 2], [4, 3],
        #                         [4, 4]])
        p = Problem(ip_width, ip_height)
        solution, feasible = p.solve(_print=True)
        # solution = [[0, 1], [1, 1]]

        if show_intersections:
            intersect_matrix, n = get_intersect_matrix(solution, allow_intersect_at_stubs=False)
        else:
            intersect_matrix = np.zeros_like(solution)

        solution_flat = np.ravel(solution, order='F')
        intersect_matrix_flat = np.ravel(intersect_matrix, order='F')
        # solution_flat = [0, 1, 0, 1, 1, 1, 0, 1, 0]
        # solution_flat = [0, 1, 1, 0]
        squares = []
        captions = []
        for i in range(num_elements):
            x, y = (i % ip_width, int(np.floor(i / ip_width)))

            coords = helper.get_element_coords(i)
            if solution_flat[i] > 0:
                if intersect_matrix_flat[i] > 0:
                    square = get_square(coords, square_size, YELLOW_E, DARK_GREY, border_width=2 * square_size)
                else:
                    square = get_square(coords, square_size, GREEN_E, DARK_GREY, border_width=2 * square_size)

                # if x == 1 and y == 1:
                squares.append(square)
                # else:
                #     square = get_square(coords, square_size, BLACK, BLUE_E, border_width=2)
                #     squares.append(square)
                # captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords))
            else:
                square = get_square(coords, square_size, BLACK, DARK_GREY, border_width=2)
                squares.append(square)

            captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords))
            # captions.append(get_text('{}'.format(int(n - x - y)), coords))

        animation_sequence = []

        if show_graph:
            graph = convert_solution_to_graph(solution, shift=[-square_size / 2, -square_size / 2])
            animation_sequence += draw_graph(graph)

        animation_sequence += [
            AnimationObject('add', content=squares, bring_to_back=True),
            AnimationObject('play', content=[Create(text) for text in captions], duration=1, bring_to_front=True)
        ]

        legend = get_legend([
                (YELLOW_E, DARK_GREY, 'Possible Intersection'),
                (BLACK, DARK_GREY, 'Negative Cell'),
                (GREEN_E, DARK_GREY, 'Positive Cell'),
            ], shift=[camera_size[0] + square_size/4, square_size/2], scale=0.3)

        animation_sequence += [
            AnimationObject('add', content=legend, bring_to_front=True)
        ]

        self.play_animations(animation_sequence)

        self.wait(5)


class IPVisualization:
    def __init__(self, path_to_config, show_graph=True, show_text='names', show_intersections=True, show_all_intersections=False, show_edges=False, show_track=False,
                 show_cells=True):
        if show_text not in ['names', 'values']:
            raise ValueError('Unknown value for show_text: {}'.format(show_text))

        self.solution = get_solution_from_config(path_to_config, _print=False, allow_gap_intersections=False)
        self.problem_dict = calculate_problem_dict(self.solution, print_time=False)

        self.problem_dict = calculate_problem_dict(self.solution, print_time=False)

        self.width, self.height = [value+1 for value in np.shape(self.solution)]
        self.ip_width, self.ip_height = (self.width - 1, self.height - 1)
        self.show_graph = show_graph
        self.show_intersections = show_intersections
        self.show_all_intersections = show_all_intersections
        self.show_track = show_track
        self.show_edges = show_edges
        self.show_root = show_edges
        self.show_text = show_text
        self.show_cells = show_cells
        self.n = (self.width * self.height - 4) / 2 + 1
        self.square_size = 1
        self.num_elements = self.ip_width * self.ip_height
        self.helper = GridShowCase(self.num_elements, [self.square_size, self.square_size], spacing=[0, 0], space_ratio=[self.ip_width, self.ip_height])
        self.animation_sequence = []

        # Descriptors
        self.negative_cell_desc = (BLACK, DARK_GREY, 'Negative Cell')
        self.positive_cell_desc = (GREEN_E, DARK_GREY, 'Positive Cell')
        self.intersection_cell_desc = (YELLOW_E, DARK_GREY, 'Intersection')
        self.root_cell_desc = (DARK_BROWN, DARK_GREY, 'Root Cell')

        # Pre-declarations
        self.arrows = []
        self.captions = []
        self.legend = []
        self.graph = None
        self.squares = []

    def get_animation_sequence(self):
        if self.show_cells:
            self.add_squares()
        if self.show_edges:
            self.add_edges()
        # self.add_legend()
        self.add_pause(3)
        if self.show_graph:
            # self.remove_edges()
            self.add_graph(animate_intersections=True)
            self.add_pause(3)
        if self.show_track:
            self.remove_squares()
            self.remove_legend()
            # self.add_track()
            self.add_track_fm(colored=True)
            # self.add_track(colored=True)
        return self.animation_sequence

    def add_pause(self, duration):
        self.animation_sequence += [AnimationObject('wait', content=[], wait_after=duration)]

    def add_squares(self):
        # unpack primary and secondary colors
        pc1, sc1, _ = self.positive_cell_desc
        pc2, sc2, _ = self.negative_cell_desc
        pc3, sc3, _ = self.intersection_cell_desc
        pc4, sc4, _ = self.root_cell_desc

        # if self.show_all_intersections:
        #     intersect_matrix, n = get_intersect_matrix(self.solution, allow_intersect_at_stubs=False)
        # elif self.show_intersections:
        #     intersect_matrix = self.problem_dict['intersections']
        # else:
        #     intersect_matrix = np.zeros_like(self.solution)
        # solution_flat = np.ravel(self.solution, order='F')
        # intersect_matrix_flat = np.ravel(intersect_matrix, order='F')

        # intersection_constraints_names = [None, None, 'bottom', None, 'left', '', 'right', None, 'top', None] + [''] * 20
        # counter2 = 0

        print_2d(self.solution)
        self.squares = []
        self.captions = []
        counter = 0
        for y in range(self.ip_height):
            for x in range(self.ip_width):
                coords = self.helper.get_element_coords(counter)
                counter += 1
                if self.solution[x][y] == SolutionEntries.positive:
                    primary_color, secondary_color, _ = self.positive_cell_desc
                    square = get_square(coords, self.square_size, primary_color, secondary_color, border_width=2 * self.square_size)
                    if x == 0 and y == 0 and self.show_root:
                        primary_color, secondary_color, _ = self.root_cell_desc
                        square = get_square(coords, self.square_size, primary_color, secondary_color, border_width=2 * self.square_size)
                elif self.solution[x][y] == SolutionEntries.positive_and_intersection:
                    primary_color, secondary_color, _ = self.intersection_cell_desc
                    square = get_square(coords, self.square_size, primary_color, secondary_color, border_width=2 * self.square_size)
                elif self.solution[x][y] == SolutionEntries.negative_and_intersection:
                    primary_color, secondary_color, _ = self.root_cell_desc
                    square = get_square(coords, self.square_size, primary_color, secondary_color, border_width=2 * self.square_size)
                elif self.solution[x][y] == SolutionEntries.negative:
                    primary_color, secondary_color, _ = self.negative_cell_desc
                    square = get_square(coords, self.square_size, primary_color, secondary_color, border_width=2 * self.square_size)

                self.squares.append(square)
                if self.show_text == 'values' and self.solution[x][y] in [SolutionEntries.positive, SolutionEntries.positive_and_intersection]:
                    node_grid_values = self.problem_dict['node_grid_values']
                    self.captions.append(get_text('{}'.format(node_grid_values[x][y]), coords))
                elif self.show_text == 'names':
                    self.captions.append(get_text(r'$c_{' + str(x+1) + ',' + str(y+1) + '}$', coords))
                    # if intersection_constraints_names[counter] is not None:
                    #    self.captions.append(get_text(r'$c_\mathrm{' + intersection_constraints_names[counter] + '}$', coords, scale=1 if intersection_constraints_names[counter] == '' else 0.65))
                    # counter2 += 1

        self.animation_sequence += [
            AnimationObject('add', content=self.squares, z_index=0),
            AnimationObject('play', content=[Create(text) for text in self.captions], duration=1, z_index=5),
            AnimationObject('add', content=self.captions, z_index=5)
        ]

    def remove_squares(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.squares),
        ]

    def remove_captions(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.captions),
        ]

    def add_edges(self, show_unselected=True):
        edges_out = self.problem_dict['edges_out']

        self.arrows = []
        for index in range(self.num_elements):
            x, y = (index % self.ip_width, int(np.floor(index / self.ip_width)))
            coords = self.helper.get_element_coords(index)
            edge_list = edges_out[(x, y)]

            for edge in edge_list:
                value, name = edge
                coords1, coords2 = name[1:].split('to')
                coords1 = np.array([int(elem) for elem in coords1.split('_')])
                coords2 = np.array([int(elem) for elem in coords2.split('_')])
                if show_unselected:
                    coords1, coords2 = shift_edge((coords1, coords2))
                coords1, coords2 = (np.array(list(coords1) + [0]), np.array(list(coords2) + [0]))

                if value > 0:
                    arrow = get_arrow(coords1 * self.square_size, coords2 * self.square_size, scale=0.5, color=RED)
                    self.arrows.append(arrow)
                elif show_unselected:
                    arrow = get_arrow(coords1 * self.square_size, coords2 * self.square_size, scale=0.5, color=GREY)
                    self.arrows.append(arrow)

        self.animation_sequence += [
            AnimationObject('add', content=self.arrows, z_index=10),
        ]

    def add_edges_special(self):
        self.arrows = []

        edge_list = [
            ([0, 0], [0, 1]),
            ([2, 0], [2, 1]),
            ([2, 1], [2, 0]),
        ]
        overwrite = []
        for index, edge in enumerate(edge_list):
            coords1, coords2 = shift_edge(edge)
            edge = (list(coords1) + [0], list(coords2) + [0])
            overwrite.append(edge)

        edge_list = overwrite

        for edge in edge_list:
            coords1, coords2 = edge
            arrow = get_arrow(coords1 * self.square_size, coords2 * self.square_size, scale=0.5, color=RED)
            self.arrows.append(arrow)

        self.animation_sequence += [
            AnimationObject('add', content=self.arrows, z_index=10),
        ]

    def remove_edges(self):
        self.animation_sequence += [
            AnimationObject('remove', content=[] if self.arrows is None else self.arrows, bring_to_front=True),
        ]

    def add_graph(self, animate_intersections=True):
        # grid = Grid(Graph(self.width, self.height, shift=[-self.square_size / 2, -self.square_size / 2]), square_size=self.square_size, shift=np.array([-self.square_size, -self.square_size]) * self.square_size, stroke_width=2)
        # self.animation_sequence += grid.get_animation_sequence()

        if animate_intersections and not self.show_edges:
            self.graph, intersections = convert_solution_to_graph(self.solution, scale=self.square_size,
                                                                  shift=[-self.square_size / 2, -self.square_size / 2],
                                                                  get_intersections=True, problem_dict=self.problem_dict)

            # self.animation_sequence += draw_graph(self.graph, z_index=15)
            # colors = [PURPLE_D, RED]
            #
            # for index, tour in enumerate(extract_graph_tours(self.graph, parse_broken=True)):
            #     edge_drawables = [edge.drawable for edge in tour.get_edges()]
            #     for drawable in edge_drawables:
            #         drawable.set_color(colors[index])
            #     self.animation_sequence.append(AnimationObject(type='add', content=edge_drawables, duration=0, z_index=15))

            self.animation_sequence += draw_graph(self.graph, z_index=15)
            self.animation_sequence += [AnimationObject('wait', content=[], wait_after=3)]
            self.animation_sequence += [AnimationObject('remove', content=self.captions)]
            if len(intersections) > 0:
                self.animation_sequence += make_concurrent([intersection.intersect() for intersection in intersections])
        else:
            self.graph = convert_solution_to_graph(self.solution, shift=[-self.square_size / 2, -self.square_size / 2],
                                                   problem_dict=self.problem_dict)
            self.animation_sequence += draw_graph(self.graph, z_index=15)
            # colors = [PURPLE_D, RED]
            #
            # self.animation_sequence += [AnimationObject('remove', content=self.captions)]
            # for index, tour in enumerate(extract_graph_tours(self.graph, parse_broken=True)):
            #     edge_drawables = [edge.drawable for edge in tour.get_edges()]
            #     for drawable in edge_drawables:
            #         drawable.set_color(colors[index])
            #     self.animation_sequence.append(AnimationObject(type='add', content=edge_drawables, duration=0, z_index=15))

    def add_track(self, track_width=0.2, colored=True):
        grid = Grid(self.graph, square_size=self.square_size, shift=np.array([-1, -1]) * self.square_size, stroke_width=1.5)
        animations_list = [grid.get_animation_sequence()]

        graph_tours = extract_graph_tours(self.graph)
        for idx, graph_tour in enumerate(graph_tours):
            gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph_tour, track_width=track_width, z_index=20)
            if colored:
                track_colors = track_properties_to_colors(track_properties)
            else:
                track_colors = None

            interpolation_animation = get_interpolation_animation_piece_wise(points, colors=track_colors, z_index=15)
            # interpolation_animation = get_interpolation_animation_continuous(points, _color=colors[idx], stroke_width=2.5)

            animations_list += [
                grid.get_animation_sequence(),
                gen_track_points,
                remove_graph(self.graph, animate=True),
                interpolation_animation,
                remove_track_points,
            ]

        animations_list.append(remove_graph(self.graph, animate=True))
        for animations in animations_list:
            self.animation_sequence += animations

    def add_track_fm(self, track_width=0.2, colored=False):
        if colored:
            colored_by = 'track_property'
        else:
            colored_by = None

        # Draw grid
        grid = Grid(self.graph, square_size=self.square_size, shift=np.array([-1, -1]) * self.square_size)
        # grid = Grid(self.graph, square_size=self.square_size, shift=np.array([-1, -1]) * self.square_size, stroke_width=2)
        self.animation_sequence += grid.get_animation_sequence()

        # get track point animations
        tp_generation_anims = list()
        tp_remove_anims = list()
        graph_tours = extract_graph_tours(self.graph)
        for graph_tour in graph_tours:
            gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph_tour, track_width=track_width, z_index=20)
            tp_generation_anims.append(gen_track_points)
            tp_remove_anims.append(remove_track_points)

        # get track animations
        fm = FeatureModel(self.solution, scale=self.square_size, shift=np.array([-0.5, -0.5]) * self.square_size, problem_dict=self.problem_dict)
        anim_sequence = []
        for index, feature in enumerate(fm.features):
            animation = feature.draw(track_width=track_width, color_by=colored_by)
            if animation is not None:
                anim_sequence.append(animation)

        # draw track points
        for animations in tp_generation_anims:
            self.animation_sequence += animations

        # remove graph and draw track
        self.animation_sequence += remove_graph(self.graph, animate=True)
        self.animation_sequence += anim_sequence

        # remove track points
        for animations in tp_remove_anims:
            self.animation_sequence += animations

        # self.play_animations(anim_sequence)

    def remove_graph_edges(self):
        return [AnimationObject(type='remove', content=[edge.drawable for edge in self.graph.edges])]

    def remove_graph_nodes(self):
        return [AnimationObject(type='remove', content=[node.drawable for node in self.graph.nodes])]

    def add_legend(self):
        legend_entries = [
            self.negative_cell_desc,
            self.positive_cell_desc
        ]
        if self.show_intersections:
            legend_entries.insert(0, self.intersection_cell_desc)
        if self.show_root:
            legend_entries.insert(0, self.root_cell_desc)

        _, camera_size, _ = self.get_camera_settings()
        self.legend = get_legend(legend_entries, shift=[camera_size[0] + self.square_size / 4, self.square_size / 2], scale=self.square_size * 0.5)
        self.animation_sequence += [
            AnimationObject('add', content=self.legend, bring_to_front=True)
        ]

    def remove_legend(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.legend)
        ]

    def get_camera_settings(self):
        camera_position, camera_size = self.helper.get_global_camera_settings()
        shift = [-self.square_size / 2, -self.square_size / 2]
        return camera_position, camera_size, shift


def get_legend(legend_list, shift, scale=1.0):
    drawables = []
    helper = GridShowCase(len(legend_list) * 2, [scale, scale], spacing=[scale, scale/2], space_ratio=[2, len(legend_list)], shift=shift)
    for index, (_color, _secondary_color, label) in enumerate(legend_list):
        element_coords = helper.get_element_coords(index * 2)
        text_coords = helper.get_element_coords(index * 2 + 1)
        drawables += [
            get_square(element_coords, scale, _color, _secondary_color, border_width=2 * scale),
            get_text(label, (text_coords[0] + len(label) * scale * 0.05, text_coords[1]), scale=scale)
        ]

    return drawables


def shift_edge(edge, shift_from_center=0.12):
    coords1, coords2 = [np.array(coords) for coords in edge]
    difference = coords1 - coords2
    if difference[0] > 0:
        print("Left")
        x, y = coords1
        coords1 = (x, y + shift_from_center)
        x, y = coords2
        coords2 = (x, y + shift_from_center)
    elif difference[0] < 0:
        print("Right")
        x, y = coords1
        coords1 = (x, y - shift_from_center)
        x, y = coords2
        coords2 = (x, y - shift_from_center)
    elif difference[1] > 0:
        print("Below")
        x, y = coords1
        coords1 = (x + shift_from_center, y)
        x, y = coords2
        coords2 = (x + shift_from_center, y)
    elif difference[1] < 0:
        print("Above")
        x, y = coords1
        coords1 = (x - shift_from_center, y)
        x, y = coords2
        coords2 = (x - shift_from_center, y)

    return coords1, coords2


class IPExtra(AnimationSequenceScene):
    def construct(self):
        path_to_config = os.path.join(os.getcwd(), 'ip/configs/innter_outer.txt')
        viz = IPVisualization(path_to_config, show_text='names', show_edges=False, show_track=True, show_graph=True, show_cells=False)
        camera_position, camera_size, shift = viz.get_camera_settings()
        # self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.5, shift=shift)
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=2, shift=shift)
        self.play_animations(viz.get_animation_sequence())
        self.wait(5)


if __name__ == '__main__':
    scene = IPExtra()
    scene.construct()
