from manim import *
from anim_sequence import AnimationSequenceScene, AnimationObject
from graph import Graph, custom_joins
from ip.ip_util import QuantityConstraint, ConditionTypes, QuantityConstraintStraight
from ip.iteration import GraphModel, get_custom_solution, convert_solution_to_graph
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous
from util import Grid, draw_graph, remove_graph, make_unitary, add_graph, generate_track_points, draw_ip_solution, TrackProperties, track_properties_to_colors, get_line
from fm.model import FeatureModel


class MultiGraphIP(AnimationSequenceScene):
    def construct(self):
        show_ip = False
        show_graph = False
        show_track = True

        width, height = (4, 4)
        square_size = 1.5
        graph_model = GraphModel(width, height, generate_intersections=False, sample_random=None)
        graph_list, helper = graph_model.get_graphs(scale=square_size, spacing=[2, 2], ratio=[16, 9])
        camera_position, camera_size = helper.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.1)

        if show_ip:
            animations = []
            for index, ip_solution in enumerate(graph_model.variants):
                shift = helper.get_element_coords(index)
                animations.append(draw_ip_solution(ip_solution, square_size, shift))
            self.play_concurrent(animations)
            self.wait(3)

        if show_graph:
            animations = [add_graph(graph, z_index=15) for graph in graph_list]
            self.play_concurrent(animations)
            self.wait(3)

        if show_track:
            track_animations_list = []  # Animations for each graph
            track_width = 0.4
            for graph in graph_list:
                gen_track_points, remove_track_points, points, _ = generate_track_points(graph, track_width=track_width)
                interpolation_animation = get_interpolation_animation_continuous(points)
                # animations = gen_track_points + interpolation_animation + remove_graph(graph, animate=True) + remove_track_points
                animations = gen_track_points + interpolation_animation + remove_track_points
                track_animations_list.append(animations)
            self.play_concurrent(track_animations_list)
            self.wait(3)


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
        width, height = (4, 4)
        square_size = 1
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
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size, stroke_width=0.6)

        graph_creation = draw_graph(graph, z_index=15)
        fade_out_non_unitary = make_unitary(graph)
        show_cycles = graph.init_cycles()
        make_joins = custom_joins(graph)
        # make_joins = random_joins(graph)
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
            remove_graph(graph, animate=True),
            interpolation_animation,
            remove_track_points,
        ]

        for anim in animations_list:
            self.play_animations(anim)

        self.wait(4)


class FMTrack(AnimationSequenceScene):
    def construct(self):
        square_size, track_width = (1, 0.2)
        width, height = (8, 8)

        solution, _ = get_custom_solution(width, height, quantity_constraints=[
            QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, quantity=0),
            QuantityConstraint(TrackProperties.turn_180, ConditionTypes.more_or_equals, quantity=0),
            QuantityConstraint(TrackProperties.turn_90, ConditionTypes.more_or_equals, quantity=0),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.equals, length=2, quantity=1),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.equals, length=3, quantity=1),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.equals, length=4, quantity=1),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.equals, length=5, quantity=1),
            QuantityConstraintStraight(TrackProperties.straight, ConditionTypes.equals, length=6, quantity=1),
        ])
        fm = FeatureModel(solution, scale=square_size)

        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))
        grid = Grid(Graph(width=width, height=height), square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)
        self.play_animations(grid.get_animation_sequence())

        anim_sequence = []

        for index, feature in enumerate(fm.features):
            # Draw Start in different color
            # if index == 0:
            #     animation = feature.draw(track_width=track_width, color_overwrite=DARK_BROWN)
            #     anim_sequence.append(animation)
            #     continue

            animation = feature.draw(track_width=track_width)
            if animation is not None:
                anim_sequence.append(animation)

            # Draw Entry Points of Intersections
            # if feature.track_property is TrackProperties.intersection:
            #     for succesor in feature.successor:
            #         line = get_line(list(feature.center[0]) + [0], succesor.end.coords, stroke_width=1.0, color=PINK)
            #         anim_sequence.append(AnimationObject('add', content=[line]))
            #     for pre_index, predecessor in enumerate(feature.predecessor):
            #         line = get_line(list(feature.center[0]) + [0], predecessor.start.coords, stroke_width=1.0, color=LIGHT_BROWN)
            #         anim_sequence.append(AnimationObject('add', content=[line]))
            #         line = get_line(predecessor.end.coords, np.array(predecessor.end.coords) + np.array(list(predecessor.end.direction) + [0]) / 4,
            #                         stroke_width=2.0, color=[RED, BLUE][pre_index])
            #         anim_sequence.append(AnimationObject('add', content=[line]))
        self.play_animations(anim_sequence)

        self.wait(4)


if __name__ == '__main__':
    # scene = CustomTrack()
    scene = FMTrack()
    scene.construct()
