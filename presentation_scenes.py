from manim import *
import random
from interpolation import find_polynomials
from iteration.ip_iteration import get_problem, get_intersect_matrix, convert_solution_to_join_sequence, GraphModel
from util import Converter, Grid, TrackPoint, GridShowCase, draw_graph, remove_graph, make_unitary, print_2d, get_square
from graph import Graph, GraphSearcher
from anim_sequence import AnimationObject, AnimationSequenceScene


class LineTest(MovingCameraScene):
    def construct(self):
        px, py = find_polynomials(0, 0, 1, 0, 0.1, 0.1, 0, 1)
        _line_x = ParametricFunction(function=lambda t: (t - 2, px(t), 0), t_min=0, t_max=2, color=WHITE)
        _line_y = ParametricFunction(function=lambda t: (t, py(t), 0), t_min=0, t_max=2, color=WHITE)
        _line = ParametricFunction(function=lambda t: (px(t) + 2, py(t), 0), t_min=0, t_max=0.1, color=WHITE)
        label_x = Text("fx(z)")
        label_x.next_to(_line_x, DOWN)
        label_y = Text("fy(z)")
        label_y.next_to(_line_y, DOWN)
        label = Text("fx,y(z)")
        label.next_to(_line, DOWN)
        # self.play(Create(label_x), Create(label_y), Create(label))
        self.add(label_x, label_y, label)
        self.play(Create(_line_x))
        self.play(Create(_line_y))
        self.play(Create(_line), run_time=2)
        self.wait(5)


class CircleTest(MovingCameraScene):
    def construct(self):
        # scale_list = [0.1, 0.5, 0.75, 1, 2, 10, 100]
        scale_list = [0.1, 0.5, 0.75, 1, 2]
        x_pos = 0
        for scale_idx, scale in enumerate(scale_list):
            circle_points = [np.array([0, 0]) * scale, np.array([1, 1]) * scale, np.array([0, 2]) * scale, np.array([-1, 1]) * scale]
            circle_directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            width = scale * 5 + 1 if scale_idx > 0 else 2.2
            spacing = 0 if scale_idx > 0 else -width * 0.9 + 1
            current_pos = x_pos + width / 2
            self.play(
                self.camera.frame.animate.set_width(width),
                run_time=0.5
            )
            self.play(
                self.camera.frame.animate.move_to((current_pos, scale / 1.5, 0)),
                run_time=1
            )
            label = Text("r={}".format(scale), size=0.5)
            label.next_to((current_pos, 0, 0), DOWN)
            self.add(label)

            x_pos += spacing + width

            for idx in range(len(circle_points)):
                next_idx = (idx + 1) % len(circle_points)
                point1, direction1 = (circle_points[idx], circle_directions[idx])
                point2, direction2 = (circle_points[next_idx], circle_directions[next_idx])
                px, py = find_polynomials(*point1, *direction1, *point2, *direction2)
                _line = ParametricFunction(function=lambda t: (current_pos + px(t), py(t), 0), t_min=0, t_max=1, color=WHITE,
                                           stroke_width=2 if scale < 10 else (np.log10(scale) + 2) * 5)
                self.play(Create(_line), run_time=0.5)

        self.wait(3)


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
        width, height = (3, 3)
        square_size = 1
        # track_width = 0.4

        ip_width, ip_height = (width - 1, height - 1)
        num_elements = ip_width * ip_height
        helper = GridShowCase(num_elements, [square_size, square_size], spacing=[0, 0], space_ratio=[1, 1])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1, shift=[-square_size/2, -square_size/2])

        problem = get_problem(width, height)
        solution, status = problem.solve()
        solution_flat = np.ravel(solution, order='F')

        squares = []
        captions = []
        for i in range(num_elements):
            x, y = (i % ip_width, int(np.floor(i / ip_width)))

            coords = helper.get_element_coords(i)
            if solution_flat[i] > 0:
                square = get_square(coords, square_size, BLUE, BLUE_E, border_width=1)
                squares.append(square)

            captions.append(get_text(r'$c_{' + str(x) + ',' + str(y) + '}$', coords))

        animation_sequence = [
            AnimationObject('add', content=squares, bring_to_back=True),
            AnimationObject('play', content=[Create(text) for text in captions], duration=1, bring_to_front=True)
        ]

        self.play_animations(animation_sequence)


        # TODO: draw graph
        # join_sequence = convert_solution_to_join_sequence(solution)
        # animation, graph = join_sequence.get_animations(scale=square_size, shift=[0, 0])
        # self.play_animations(draw_graph(graph))
        self.wait(5)


def get_text(text, coords, scale=1):
    x, y = coords
    text = Tex(text).scale(scale)
    text.set_x(coords[0])
    text.set_y(coords[1])
    return text


if __name__ == '__main__':
    scene = Basics()
    scene.construct()
