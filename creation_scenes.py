import os

from manim import *
from termcolor import colored
from tqdm import tqdm

from anim_sequence import AnimationSequenceScene, AnimationObject
from graph import Graph, custom_joins
from ip.ip_util import QuantityConstraint, ConditionTypes, QuantityConstraintStraight, parse_ip_config
from ip.iteration import GraphModel, get_custom_solution, convert_solution_to_graph, get_solution_from_config, ZoneDescription, get_zone_solution
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous
from ip.problem import Problem
from util import Grid, draw_graph, remove_graph, make_unitary, add_graph, generate_track_points, draw_ip_solution, TrackProperties, track_properties_to_colors, get_line, \
    extract_graph_tours, ZoneTypes
from fm.model import FeatureModel


class MultiGraphIP(AnimationSequenceScene):
    def construct(self):
        show_ip = False
        show_graph = False
        show_track = True

        width, height = (4, 4)
        square_size = 1
        graph_model = GraphModel(width, height, generate_intersections=False, sample_random=None)
        graph_list, helper = graph_model.get_graphs(scale=square_size, spacing=[1, 1], ratio=[6, 1])
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
                graph_tours = extract_graph_tours(graph)
                gen_track_points, remove_track_points, points, _ = generate_track_points(graph_tours[0], track_width=track_width)
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
        track_width = 0.2
        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))

        # solution = get_random_solution(width, height)
        # solution, _ = get_custom_solution(width, height)
        path_to_config = os.path.join(os.getcwd(), 'ip/configs/mini_no_intersect.txt')
        solution = get_solution_from_config(path_to_config)

        # Animate Solution
        graph = convert_solution_to_graph(solution, scale=square_size)
        graph_tours = extract_graph_tours(graph)
        gen_track_points, remove_track_points, points, _ = generate_track_points(graph_tours[0], track_width=track_width)
        # interpolation_animation = get_interpolation_animation_continuous(points)
        interpolation_animation = get_interpolation_animation_piece_wise(points)
        grid = Grid(graph, square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)

        animations_list = [
            grid.get_animation_sequence(),
            draw_graph(graph, z_index=0),
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
        graph_tours = extract_graph_tours(graph)
        gen_track_points, remove_track_points, points, _ = generate_track_points(graph_tours[0], track_width=track_width)
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
        graph_tours = extract_graph_tours(graph)
        gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph_tours[0], track_width=track_width)
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
        square_size, track_width = (1, 0.266)
        path_to_config = os.path.join(os.getcwd(), 'ip/configs/plain.txt')
        anim_fm = False
        show_graph = True

        solution = get_solution_from_config(path_to_config, _print=False)
        width, height = [value+1 for value in np.shape(solution)]
        fm = FeatureModel(solution, scale=square_size)

        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))
        grid = Grid(Graph(width=width, height=height), square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size, stroke_width=1.5)
        self.play_animations(grid.get_animation_sequence(z_index=20))

        anim_sequence = []

        if anim_fm:
            for index, feature in enumerate(fm.features):
                animation = feature.draw(track_width=track_width, color_by='track_property')
                if animation is not None:
                    anim_sequence.append(animation)
            self.play_animations(anim_sequence)
        else:
            if show_graph:
                self.play_animations(add_graph(fm.graph, z_index=25))
            else:
                graph_tours = extract_graph_tours(fm.graph)
                colored_by_properties = False
                colors = [YELLOW, BLUE_C, GREEN, ORANGE, PINK, PURPLE]
                for index, graph_tour in enumerate(graph_tours):
                    gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph_tour, track_width=track_width, z_index=20)
                    if colored_by_properties:
                        track_colors = track_properties_to_colors(track_properties)
                    else:
                        track_colors = [colors[index] for _ in track_properties]

                    # interpolation_animation = get_interpolation_animation_piece_wise(points, colors=track_colors, z_index=15)
                    interpolation_animation = get_interpolation_animation_continuous(points)

                    anim_sequence += [
                        interpolation_animation,
                    ]
                    print(colored("Rendering...", 'cyan'))
                    for animations in tqdm(anim_sequence, desc="rendering"):
                        self.play_animations(animations)

        self.wait(4)


class FMTrackZones(AnimationSequenceScene):
    def construct(self):
        square_size, track_width = (1, 0.2)
        anim_fm = True
        path_to_config = os.path.join(os.getcwd(), 'ip/configs/demo.txt')

        zone_descriptions = [
            ZoneDescription(ZoneTypes.motorway, min_length=3, max_length=4),
            # ZoneDescription(ZoneTypes.motorway, min_length=4, max_length=4),
            ZoneDescription(ZoneTypes.urban_area, min_length=3, max_length=5),
            # ZoneDescription(ZoneTypes.urban_area, min_length=6, max_length=10),
            # ZoneDescription(ZoneTypes.no_passing, min_length=6, max_length=6),
            # ZoneDescription(ZoneTypes.no_passing, min_length=6, max_length=6),
            # ZoneDescription(ZoneTypes.no_passing, min_length=6, max_length=6),
        ]
        solution, zone_selection = get_zone_solution(path_to_config, zone_descriptions)
        fm = FeatureModel(solution, zone_selection, scale=1)
        width, height = [value + 1 for value in np.shape(solution)]

        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))
        grid = Grid(Graph(width=width, height=height), square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)
        self.play_animations(grid.get_animation_sequence())

        anim_sequence = []

        if anim_fm:
            for index, feature in enumerate(fm.features):
                animation = feature.draw(track_width=track_width, color_by='zone')
                if animation is not None:
                    anim_sequence.append(animation)
            self.play_animations(anim_sequence)
        else:
            graph_tours = extract_graph_tours(fm.graph)
            colored_by_properties = False
            colors = [YELLOW, BLUE_C, GREEN, ORANGE, PINK, PURPLE]
            for index, graph_tour in enumerate(graph_tours):
                gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph_tour, track_width=track_width, z_index=20)
                if colored_by_properties:
                    track_colors = track_properties_to_colors(track_properties)
                else:
                    track_colors = [colors[index] for _ in track_properties]

                interpolation_animation = get_interpolation_animation_piece_wise(points, colors=track_colors, z_index=15)
                # interpolation_animation = get_interpolation_animation_continuous(points)

                anim_sequence += [
                    interpolation_animation,
                ]
                print(colored("Rendering...", 'cyan'))
                for animations in tqdm(anim_sequence, desc="rendering"):
                    self.play_animations(animations)

        self.wait(4)


class CC20(AnimationSequenceScene):
    def construct(self):
        square_size, track_width = (1, 0.15)

        original = [[1, 1, 1], [2, 0, 1], [1, 0, 1], [2, 0, 1], [1, 3, 1], [2, 0, 1], [1, 0, 1]]
        p = Problem(7, 3, imitate=original, allow_gap_intersections=True, allow_adjacent_intersections=True,
                    quantity_constraints=[QuantityConstraint(TrackProperties.intersection, ConditionTypes.more_or_equals, quantity=0)])
        solution, status = p.solve(_print=True)
        width, height = [value+1 for value in np.shape(solution)]
        fm = FeatureModel(solution, scale=square_size)

        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))
        grid = Grid(Graph(width=width, height=height), square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size)
        self.play_animations(grid.get_animation_sequence())

        anim_sequence = []
        for index, feature in enumerate(fm.features):
            animation = feature.draw(track_width=track_width)
            if animation is not None:
                anim_sequence.append(animation)
        self.play_animations(anim_sequence)
        self.wait(4)


class MultiGraphFM(AnimationSequenceScene):
    def construct(self):
        show_ip = False
        show_graph = False
        show_track = True
        colored_track = False

        width, height = (4, 4)
        square_size = 1.5
        graph_model = GraphModel(width, height, allow_gap_intersections=True, allow_adjacent_intersections=False, sample_random=None)
        graph_list, helper = graph_model.get_graphs(scale=square_size, spacing=[2, 2], ratio=[16, 9])
        meta_info = graph_model.get_meta_info()
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
            for index, graph in tqdm(enumerate(graph_list), desc="drawing tracks"):
                track_info = meta_info[index]
                graph_tours = extract_graph_tours(graph)
                for graph_tour in graph_tours:
                    gen_track_points, remove_track_points, points, track_properties = generate_track_points(graph_tour, track_width=track_width, z_index=20)
                    if colored_track:
                        track_colors = track_properties_to_colors(track_properties)
                    else:
                        if track_info.number_gap_intersections > 0:
                            track_colors = [YELLOW for _ in track_properties]
                        else:
                            track_colors = None

                    interpolation_animation = get_interpolation_animation_piece_wise(points, colors=track_colors, z_index=15)
                    # interpolation_animation = get_interpolation_animation_continuous(points)

                    track_animations_list += [
                        # gen_track_points,
                        remove_graph(graph, animate=False),
                        interpolation_animation,
                        # remove_track_points,
                    ]

            print(colored("Rendering...", 'cyan'))
            for animations in tqdm(track_animations_list, desc="rendering"):
                self.play_animations(animations)
            # self.play_concurrent(track_animations_list)
            self.wait(3)


if __name__ == '__main__':
    # scene = FMTrack()
    scene = MultiGraphFM()
    scene.construct()
