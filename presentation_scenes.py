from manim import *
import random
from interpolation import find_polynomials
from ip import GGMSTProblem
from iteration.ip_iteration import get_problem, get_intersect_matrix, convert_solution_to_join_sequence, GraphModel, convert_solution_to_graph, \
    get_custom_solution
from util import GraphTour, Grid, TrackPoint, GridShowCase, draw_graph, remove_graph, make_unitary, print_2d, get_square, get_text
from graph import Graph, GraphSearcher
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
        p = GGMSTProblem(ip_width, ip_height)
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


if __name__ == '__main__':
    scene = IP()
    scene.construct()
