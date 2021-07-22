from manim import *
from anim_sequence import AnimationSequenceScene
from graph import Graph, random_joins, custom_joins
from ip.ip_util import QuantityConstraint, TrackProperties, ConditionTypes
from ip.iteration import GraphModel, get_custom_solution, get_random_solution, convert_solution_to_graph
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous
from util import Grid, draw_graph, remove_graph, make_unitary, add_graph, generate_track_points


class MultiGraphIP(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 2
        graph_model = GraphModel(width, height, generate_intersections=False, sample_random=None)
        graph_list, helper = graph_model.get_graphs(scale=square_size, spacing=[2, 2], ratio=[6, 4])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1)
        animations = [add_graph(graph) for graph in graph_list]
        self.play_concurrent(animations)
        self.wait(3)

        track_animations_list = []  # Animations for each graph
        track_width = 0.4
        for graph in graph_list:
            gen_track_points, remove_track_points, points, _ = generate_track_points(graph, track_width=track_width)
            interpolation_animation = get_interpolation_animation_continuous(points)
            animations = gen_track_points + interpolation_animation + remove_graph(graph, animate=True) + remove_track_points
            track_animations_list.append(animations)

        self.play_concurrent(track_animations_list)
        self.wait(5)


class IPCircuitCreation(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1
        track_width = 0.1
        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))

        # solution = get_random_solution(width, height)
        solution, _ = get_custom_solution(width, height)

        # Animate Solution
        graph = convert_solution_to_graph(solution, scale=square_size)
        gen_track_points, remove_track_points, points, _ = generate_track_points(graph, track_width=track_width)
        # interpolation_animation = get_interpolation_animation_continuous(points)
        interpolation_animation = get_interpolation_animation_piece_wise(points)
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)

        animations_list = [
            grid.get_animation_sequence(),
            draw_graph(graph),
            gen_track_points,
            remove_graph(graph, animate=True),
            interpolation_animation,
            remove_track_points,
        ]

        for anim in animations_list:
            self.play_animations(anim)

        self.wait(4)


class CircuitCreation(AnimationSequenceScene):
    def construct(self):
        width, height = (6, 6)
        square_size = 2
        track_width = 0.3

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

        graph_creation = draw_graph(graph)
        fade_out_non_unitary = make_unitary(graph)
        show_cycles = graph.init_cycles()
        # make_joins = custom_joins(graph)
        make_joins = random_joins(graph)
        gen_track_points, remove_track_points, points, _ = generate_track_points(graph, track_width=track_width)
        interpolation_animation = get_interpolation_animation_continuous(points)

        animations_list = [
            graph_creation,
            grid.get_animation_sequence(),
            fade_out_non_unitary,
            make_joins,
            gen_track_points,
            interpolation_animation,
            remove_track_points,
            remove_graph(graph)
        ]

        for animations in animations_list:
            self.play_animations(animations)

        self.wait(5)


class MultiGraph(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        square_size = 1
        graph_model = GraphModel(width, height, generate_intersections=False, fast=False)
        animations_list, graph_list, helper = graph_model.get_animations(scale=square_size, spacing=[2, 2])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1)
        animations = [add_graph(graph) for graph in graph_list]
        self.play_concurrent(animations)
        self.wait(5)


def track_properties_to_colors(track_properties):
    colors = []
    property_to_color = {
        TrackProperties.turn_90: PINK,
        TrackProperties.turn_180: GREEN,
        TrackProperties.straight: RED,
        TrackProperties.intersection: BLUE_C
    }
    for track_property in track_properties:
        if track_property is None:
            colors.append(WHITE)
        else:
            colors.append(property_to_color[track_property])

    return colors


class CustomTrack(AnimationSequenceScene):
    def construct(self):
        quantity_constraints = [
            QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, 0),
            QuantityConstraint(TrackProperties.straight, ConditionTypes.more_or_equals, 1),
            QuantityConstraint(TrackProperties.turn_180, ConditionTypes.more_or_equals, 0),
            QuantityConstraint(TrackProperties.turn_90, ConditionTypes.more_or_equals, 0)
        ]
        width, height = (8, 8)
        # square_size, track_width = (2, 0.2)
        square_size, track_width = (2, 0.4)
        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))

        # solution = get_random_solution(width, height)
        # solution = get_custom_solution(width, height, quantity_constraints=quantity_constraints)
        solution, problem_dict = get_custom_solution(width, height, quantity_constraints=quantity_constraints, iteration_constraints=[])

        # Animate Solution
        graph = convert_solution_to_graph(solution, problem_dict=problem_dict, scale=square_size)
        gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph, track_width=track_width)
        track_colors = track_properties_to_colors(track_properties)
        interpolation_animation = get_interpolation_animation_piece_wise(points, colors=track_colors)
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)

        animations_list = [
            grid.get_animation_sequence(),
            draw_graph(graph),
            gen_track_points,
            # remove_graph(graph, animate=True),
            interpolation_animation,
            remove_track_points,
        ]

        for anim in animations_list:
            self.play_animations(anim)

        self.wait(4)


if __name__ == '__main__':
    scene = CustomTrack()
    scene.construct()
