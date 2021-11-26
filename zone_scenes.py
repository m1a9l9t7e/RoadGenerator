import os
import sys

from manim import *
from termcolor import colored
from tqdm import tqdm

from anim_sequence import AnimationSequenceScene, AnimationObject
from config_parser import Config
from graph import Graph, custom_joins
from ip.ip_util import QuantityConstraint, ConditionTypes, QuantityConstraintStraight, parse_ip_config
from ip.iteration import GraphModel, get_custom_solution, convert_solution_to_graph, get_solution_from_config, ZoneDescription, get_zone_solution, IteratorType, get_zone_assignment
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous
from ip.problem import Problem
from util import Grid, draw_graph, remove_graph, make_unitary, add_graph, generate_track_points, draw_ip_solution, TrackProperties, track_properties_to_colors, get_line, \
    extract_graph_tours, ZoneTypes, get_arrow, get_text
from fm.model import FeatureModel


def zones_to_color(zones):
    if ZoneTypes.urban_area in zones:
        return ORANGE
    elif ZoneTypes.express_way in zones:
        return BLUE_C
    elif ZoneTypes.parking in zones:
        return GREEN
    elif ZoneTypes.no_passing in zones:
        return RED
    else:
        return WHITE


class FMTrackZones(AnimationSequenceScene):
    def construct(self):
        scale, track_width = (1, 0.21)
        color_zone_start_end = False
        draw_track = True
        draw_nl = not draw_track
        config = Config('/home/malte/PycharmProjects/circuit-creator/super_configs/zones.json')

        zone_descriptions = [
            ZoneDescription(ZoneTypes.parking, min_length=3, max_length=3),
            ZoneDescription(ZoneTypes.express_way, min_length=4, max_length=4),
            ZoneDescription(ZoneTypes.urban_area, min_length=8, max_length=10),
            ZoneDescription(ZoneTypes.no_passing, min_length=3, max_length=6),
        ]

        ip_solution, problem_dict = config.get_layout()
        w, h = [value + 1 for value in np.shape(ip_solution)]
        description_length = w * h
        zone_assignment, start_index = config.get_zones(ip_solution, problem_dict)
        raw = dict()
        for key in zone_assignment.keys():
            arrays = np.array(zone_assignment[key]) - start_index
            zones = []
            for array in arrays:
                start, end = array
                start = start % description_length
                end = end % description_length
                zones.append([start, end])
            raw[key] = zones

        # zone_selection, start_index, raw, description_length = get_zone_assignment(ip_solution, zone_descriptions, problem_dict, get_raw=True)

        print("RAW Zone Assignment: Parking: {}, Expressways: {}, Urban Areas: {}, No Passing: {}".format(colored(raw[ZoneTypes.parking], 'cyan'),
                                                                                                          colored(raw[ZoneTypes.express_way], 'blue'),
                                                                                                          colored(raw[ZoneTypes.urban_area], 'yellow'),
                                                                                                          colored(raw[ZoneTypes.no_passing], 'red')))
        if draw_nl:
            height = 2
            self.move_camera((scale * description_length * 1.1, scale * height), (scale * description_length / 2, scale * height / 2), resolution=[int(description_length * 1.1), height])

            anim_sequence = []

            for idx, zone_type in enumerate([ZoneTypes.parking, ZoneTypes.express_way, ZoneTypes.urban_area, ZoneTypes.no_passing]):
                zones = raw[zone_type]
                for (start, end) in zones:
                    zone_color = zones_to_color([zone_type])
                    zone_number_line = NumberLine(
                        x_range=[start, end, end-start],
                        color=zone_color,
                        unit_size=scale,
                        font_size=28,
                        stroke_width=scale * 4
                    )
                    x_shift = start + (end - start) / 2
                    zone_number_line.set_x(x_shift * scale)
                    zone_number_line.set_y(scale)
                    anim_sequence.append(AnimationObject(type='add', content=zone_number_line, z_index=10))

            description_length -= 1
            base_number_line = NumberLine(
                x_range=[0, description_length, 1],
                unit_size=scale,
                # numbers_with_elongated_ticks=[-2, 4],
                include_numbers=True,
                font_size=36,
                stroke_width=scale * 4
            )

            x_shift = description_length / 2
            base_number_line.set_x(x_shift * scale)
            base_number_line.set_y(0)
            # num6 = base_number_line.numbers[2]
            # num6.set_color(RED)

            anim_sequence.append(AnimationObject(type='add', content=base_number_line))

        else:
            fm = FeatureModel(ip_solution, zone_assignment, scale=1, start_index=start_index)
            # fm = FeatureModel(solution, zone_selection=None, scale=1, start_index=start_index)
            print("Track start at: {}".format(fm.start))
            width, height = [value + 1 for value in np.shape(ip_solution)]

            self.move_camera((scale * width * 1.1, scale * height * 1.1), (scale * width / 2.5, scale * height / 2.5, 0))
            grid = Grid(Graph(width=width, height=height), square_size=scale, shift=np.array([-0.5, -0.5]) * scale, stroke_width=0.4)
            self.play_animations(grid.get_animation_sequence())

            anim_sequence = []

            for index, feature in enumerate(fm.features):
                zone_start, zone_type = feature.is_zone_start()
                zone_end, zone_type = feature.is_zone_end()
                if zone_start and color_zone_start_end:
                    animation = feature.draw(track_width=track_width, color_by=GREEN)
                elif zone_end and color_zone_start_end:
                    animation = feature.draw(track_width=track_width, color_by=RED)
                else:
                    animation = feature.draw(track_width=track_width, color_by='zone')
                    # animation = feature.draw(track_width=track_width, color_by='track_property')
                if animation is not None:
                    anim_sequence.append(animation)

                if index == start_index:
                    coords = feature.start.logical_coords()
                    x, y = coords
                    # y -= 0.21
                    arrow = get_arrow((x, y + 0.8, 0), (x, y, 0))
                    text = get_text("0", (x, y + 0.7), scale=0.5)
                    anim_sequence.append(AnimationObject(type='add', content=arrow))
                    anim_sequence.append(AnimationObject(type='add', content=text))

        self.play_animations(anim_sequence)


class NumberLineExample(Scene):
    def construct(self):
        l0 = NumberLine(
            x_range=[2, 8, 2],
            color=BLUE,
            # include_numbers=True,
            unit_size=0.5,
            font_size=24,
        )

        l1 = NumberLine(
            x_range=[0, 10, 2],
            unit_size=0.5,
            # numbers_with_elongated_ticks=[-2, 4],
            include_numbers=True,
            font_size=24,
        )
        num6 = l1.numbers[2]
        num6.set_color(RED)

        line_group = VGroup(l0, l1).arrange(DOWN, buff=1)
        self.add(line_group)


if __name__ == '__main__':
    # scene = NumberLineExample()
    scene = FMTrackZones()
    scene.construct()
