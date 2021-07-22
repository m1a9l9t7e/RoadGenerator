from manim import *
from ip.problem import Problem
from ip.iteration import get_intersect_matrix, convert_solution_to_graph, get_custom_solution
from util import Grid, GridShowCase, draw_graph, get_square, get_text, get_arrow
from graph import Graph
from anim_sequence import AnimationObject, AnimationSequenceScene


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
        width, height = (6, 6)
        scale = 1
        # track_width = 0.4
        show_graph = True
        show_intersections = False

        ip_width, ip_height = (width - 1, height - 1)
        n = np.ceil(ip_width / 2) * ip_height + np.floor(ip_width / 2)
        num_elements = ip_width * ip_height
        helper = GridShowCase(num_elements, [scale, scale], spacing=[0, 0], space_ratio=[1, 1])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1, shift=[-scale/2, -scale/2])

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
                    square = get_square(coords, scale, YELLOW_E, DARK_GREY, border_width=2 * scale)
                else:
                    square = get_square(coords, scale, GREEN_E, DARK_GREY, border_width=2 * scale)

                # if x == 1 and y == 1:
                squares.append(square)
                # else:
                #     square = get_square(coords, square_size, BLACK, BLUE_E, border_width=2)
                #     squares.append(square)
                # captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords))
            else:
                square = get_square(coords, scale, BLACK, DARK_GREY, border_width=2)
                squares.append(square)

            captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords))
            # captions.append(get_text('{}'.format(int(n - x - y)), coords))

        animation_sequence = []

        if show_graph:
            graph = convert_solution_to_graph(solution, shift=[-scale / 2, -scale / 2])
            animation_sequence += draw_graph(graph)

        animation_sequence += [
            AnimationObject('add', content=squares, bring_to_back=True),
            AnimationObject('play', content=[Create(text) for text in captions], duration=1, bring_to_front=True)
        ]

        animation_sequence += [
            get_legend([
                (YELLOW_E, DARK_GREY, 'Possible Intersection'),
                (BLACK, DARK_GREY, 'Negative Cell'),
                (GREEN_E, DARK_GREY, 'Positive Cell'),
            ], shift=[camera_size[0] + scale/4, scale/2], scale=0.3)
        ]

        self.play_animations(animation_sequence)

        self.wait(5)


class IPVisualization:
    def __init__(self, width, height, show_graph=True, show_text='names', show_intersections=True, show_edges=False, show_straights=False):
        if show_text not in ['names', 'values']:
            raise ValueError('Unknown value for show_text: {}'.format(show_text))
        self.width, self.height = (width, height)
        self.ip_width, self.ip_height = (width - 1, height - 1)
        self.show_graph = show_graph
        self.show_intersections = show_intersections
        self.show_straights = show_straights
        self.show_edges = show_edges
        self.show_root = show_edges
        self.show_text = show_text
        self.n = np.ceil(self.ip_width / 2) * self.ip_height + np.floor(self.ip_width / 2)
        self.scale = 1
        self.num_elements = self.ip_width * self.ip_height
        self.helper = GridShowCase(self.num_elements, [self.scale, self.scale], spacing=[0, 0], space_ratio=[1, 1])
        self.problem = Problem(self.ip_width, self.ip_height)
        self.solution, self.feasible = self.problem.solve(_print=True)
        # self.problem.print_all_variables()
        self.problem_dict = self.problem.export_variables()
        self.animation_sequence = []

        # Descriptors
        self.negative_cell_desc = (BLACK, DARK_GREY, 'Negative Cell')
        self.positive_cell_desc = (GREEN_E, DARK_GREY, 'Positive Cell')
        self.intersection_cell_desc = (YELLOW_E, DARK_GREY, 'Intersection')
        self.root_cell_desc = (DARK_BROWN, DARK_GREY, 'Root Cell')

    def get_animation_sequence(self):
        if self.show_graph:
            self.add_graph()
        self.add_squares()
        if self.show_edges:
            self.add_edges()
        self.add_legend()
        return self.animation_sequence

    def add_squares(self):
        # unpack primary and secondary colors
        pc1, sc1, _ = self.positive_cell_desc
        pc2, sc2, _ = self.negative_cell_desc
        pc3, sc3, _ = self.intersection_cell_desc
        pc4, sc4, _ = self.root_cell_desc

        node_grid_values = self.problem_dict['node_grid_values']

        if self.show_intersections:
            intersect_matrix, n = get_intersect_matrix(self.solution, allow_intersect_at_stubs=False)
        else:
            intersect_matrix = np.zeros_like(self.solution)

        solution_flat = np.ravel(self.solution, order='F')
        intersect_matrix_flat = np.ravel(intersect_matrix, order='F')
        squares = []
        captions = []
        for index in range(self.num_elements):
            x, y = (index % self.ip_width, int(np.floor(index / self.ip_width)))
            coords = self.helper.get_element_coords(index)
            if solution_flat[index] > 0:
                if intersect_matrix_flat[index] > 0:
                    square = get_square(coords, self.scale, pc3, sc3, border_width=2 * self.scale)
                else:
                    if x == 0 and y == 0 and self.show_root:
                        square = get_square(coords, self.scale, pc4, sc4, border_width=2 * self.scale)
                    else:
                        square = get_square(coords, self.scale, pc1, sc1, border_width=2 * self.scale)

                squares.append(square)
                if self.show_text == 'values':
                    captions.append(get_text('{}'.format(node_grid_values[x][y]), coords))
            else:
                square = get_square(coords, self.scale, pc2, sc2, border_width=2)
                squares.append(square)

            if self.show_text == 'names':
                captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords))

        self.animation_sequence += [
            AnimationObject('add', content=squares, bring_to_back=True),
            AnimationObject('play', content=[Create(text) for text in captions], duration=1, bring_to_front=True)
        ]

    def add_edges(self):
        edges_out = self.problem_dict['edges_out']

        arrows = []
        for index in range(self.num_elements):
            x, y = (index % self.ip_width, int(np.floor(index / self.ip_width)))
            coords = self.helper.get_element_coords(index)
            edge_list = edges_out[(x, y)]
            for edge in edge_list:
                value, name = edge
                coords1, coords2 = name[1:].split('to')
                coords1 = np.array([int(elem) for elem in coords1.split('_') + [0]])
                coords2 = np.array([int(elem) for elem in coords2.split('_') + [0]])
                if value > 0:
                    arrow = get_arrow(coords1 * self.scale, coords2 * self.scale, scale=0.5, color=RED)
                    arrows.append(arrow)
                # else:
                #     arrow = get_arrow(coords1 * self.scale, coords2 * self.scale, scale=0.5, color=GREY)
                #     arrows.append(arrow)

        self.animation_sequence += [
            AnimationObject('add', content=arrows, bring_to_front=True),
        ]

    def add_graph(self):
        graph = convert_solution_to_graph(self.solution, shift=[-self.scale / 2, -self.scale / 2])
        self.animation_sequence += draw_graph(graph)

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
        self.animation_sequence += [
            get_legend(legend_entries, shift=[camera_size[0] + self.scale/4, self.scale/2], scale=self.scale * 0.5)
        ]

    def get_camera_settings(self):
        camera_position, camera_size = self.helper.get_global_camera_settings()
        shift = [-self.scale/2, -self.scale/2]
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

    return AnimationObject('add', content=drawables, bring_to_front=True)


class IPExtra(AnimationSequenceScene):
    def construct(self):
        viz = IPVisualization(8, 8, show_text='values', show_edges=True, show_intersections=False)
        camera_position, camera_size, shift = viz.get_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1, shift=shift)
        self.play_animations(viz.get_animation_sequence())
        self.wait(5)


if __name__ == '__main__':
    scene = IPExtra()
    scene.construct()
