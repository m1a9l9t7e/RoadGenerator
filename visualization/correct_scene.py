import os

from manim import *
from termcolor import colored
from tqdm import tqdm

from anim_sequence import AnimationSequenceScene, AnimationObject
from config_parser import Config
from graph import Graph, custom_joins
from ip.ip_util import QuantityConstraint, ConditionTypes, QuantityConstraintStraight, parse_ip_config
from ip.iteration import GraphModel, get_custom_solution, convert_solution_to_graph, get_solution_from_config, ZoneDescription, get_zone_solution, IteratorType
from interpolation import get_interpolation_animation_piece_wise, get_interpolation_animation_continuous
from ip.problem import Problem
from util import Grid, draw_graph, remove_graph, make_unitary, add_graph, generate_track_points, draw_ip_solution, TrackProperties, track_properties_to_colors, get_line, \
    extract_graph_tours, ZoneTypes, get_text
from fm.model import FeatureModel


class TrackCorrect(AnimationSequenceScene):
    def construct(self):
        track_width = 0.42
        # track_width = 0.21
        anim_fm = True
        show_graph = False
        # show_graph = True
        text_scale = 1.2
        # path_to_config = '/home/malte/PycharmProjects/circuit-creator/super_configs/debug.json'

        # colors = [BLUE, GREEN, GOLD, MAROON, TEAL, YELLOW, RED, PURPLE]
        colors = [BLUE_D, GOLD_D, MAROON_D, RED_D, PURPLE_D]
        text_colors = [BLUE_D, GOLD_D, MAROON_D, RED_D, PURPLE_D]

        color_counter = 0
        # path_to_config = '/home/malte/PycharmProjects/circuit-creator/super_configs/zones_v0.json'
        # path_to_config = '/home/malte/PycharmProjects/circuit-creator/super_configs/layout.json'
        # path_to_config = '/home/malte/PycharmProjects/circuit-creator/super_configs/cc17.json'
        # path_to_config = '/home/malte/PycharmProjects/circuit-creator/super_configs/cc20_real.json'
        path_to_config = '/super_configs/correct.json'
        config = Config(path_to_config)
        print(config.layout.solution)
        square_size = config.layout.scale
        width, height = config.dimensions

        if show_graph:
            anim_fm = False
            square_size = 1

        self.move_camera((square_size * width * 1.1, square_size * height * 1.1), (square_size * width / 2.5, square_size * height / 2.5, 0))
        grid = Grid(Graph(width=width, height=height), square_size=square_size, shift=np.array([-0.5, -0.5]) * square_size, stroke_width=2)
        self.play_animations(grid.get_animation_sequence(z_index=-20))

        fm = config.get_fm()
        # fm = FeatureModel()
        # fm.load(config.features.fm_path)

        anim_sequence = []
        for index, feature in enumerate(fm.features):
            animation = feature.draw(track_width=track_width, color_by=WHITE, stroke_width=4)
            if animation is not None:
                anim_sequence.append(animation)
        # correctness arrows
        lane_width = track_width * 2
        tile_size = square_size
        border_size = (tile_size - lane_width) / 2

        # 0. tile size
        graph = Graph(width=width, height=height, scale=square_size, shift=[-0.5 * square_size, -0.5 * square_size])
        arrow, text_pos = get_arrow(start=graph.grid[0][0].get_real_coords(), end=graph.grid[0][1].get_real_coords(), color=colors[color_counter])
        anim_sequence.append(AnimationObject('add', arrow))

        distance = np.linalg.norm(np.array(graph.grid[0][0].get_real_coords()) - np.array(graph.grid[0][1].get_real_coords()))
        print("tile size: {:.3f}".format(distance))

        text_pos = shift_pos(text_pos, x=square_size/2.3, y=-square_size/4)
        text = get_text('{} mm'.format(to_mm(distance)), text_pos, scale=text_scale, color=text_colors[color_counter])
        anim_sequence.append(AnimationObject('add', text))
        color_counter += 1

        # 1. distance from track border
        x, y, z = graph.grid[2][1].get_real_coords()
        y -= (border_size + lane_width)
        arrow, text_pos = get_arrow(start=graph.grid[2][0].get_real_coords(), end=(x, y, z), color=colors[color_counter])
        anim_sequence.append(AnimationObject('add', arrow))

        distance = np.linalg.norm(np.array([graph.grid[2][0].get_real_coords()]) - np.array([x, y, z]))
        print("border distance: {:.3f}".format(distance))

        text_pos = shift_pos(text_pos, x=square_size/2.3)
        text = get_text('{} mm'.format(to_mm(distance)), text_pos, scale=text_scale, color=text_colors[color_counter])
        anim_sequence.append(AnimationObject('add', text))
        color_counter += 1

        # 2. minimum curve radius
        x, y, z = graph.grid[3][1].get_real_coords()
        distance = np.sqrt(border_size/2)
        arrow, text_pos = get_arrow(start=(x, y, 0), end=(x+distance, y+distance, 0), color=colors[color_counter])
        anim_sequence.append(AnimationObject('add', arrow))

        distance = np.linalg.norm(np.array([x, y, 0]) - np.array([x+distance, y+distance, 0]))
        print("minimum curve radius: {:.3f}".format(distance))

        text_pos = shift_pos(text_pos, x=-square_size/2, y=square_size/11)
        text = get_text('{} mm'.format(to_mm(distance)), text_pos, scale=text_scale, color=text_colors[color_counter])
        anim_sequence.append(AnimationObject('add', text))
        color_counter += 1

        # # 3. distance between track pieces
        x, y, z = graph.grid[2][3].get_real_coords()
        arrow, text_pos = get_arrow(start=(x, y + border_size, z), end=(x, y - border_size, z), color=colors[color_counter])
        anim_sequence.append(AnimationObject('add', arrow))

        distance = np.linalg.norm(np.array([x, y + border_size, z]) - np.array([x, y - border_size, z]))
        print("distance between: {:.3f}".format(distance))

        text_pos = shift_pos(text_pos, x=square_size/2.3, y=square_size/8)
        text = get_text('{} mm'.format(to_mm(distance)), text_pos, scale=text_scale, color=text_colors[color_counter])
        anim_sequence.append(AnimationObject('add', text))
        color_counter += 1

        self.play_animations(anim_sequence)


class GraphicsDoubleArrow(Scene):
    def construct(self):
        self.play(Write(get_arrow(start=(0, 0, 0), end=(5, 0, 0))))
        self.wait()


def get_arrow(start, end, stroke_width=6, color=WHITE):
    # double_arrow = DoubleArrow(start=start, end=end, stroke_width=stroke_width, max_stroke_width_to_length_ratio=6, max_tip_length_to_length_ratio=0.25)
    double_arrow = Line(start=start, end=end, stroke_width=stroke_width)
    double_arrow.set_color(color)
    # arrow args
    # buff = MED_SMALL_BUFF,
    # max_tip_length_to_length_ratio = 0.25,
    max_stroke_width_to_length_ratio = 1,

    text_pos = (np.array(start) + np.array(end)) / 2

    return double_arrow, text_pos[:2]


def to_mm(length):
    return int(length * 1000)


def shift_pos(pos, x=0, y=0):
    _x, _y = pos
    return np.array([_x + x, _y + y])


if __name__ == '__main__':
    scene = GraphicsDoubleArrow()
    scene.construct()
