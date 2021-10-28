import math
import sys
import xml.etree.ElementTree as ET
import numpy as np
from termcolor import colored
from anim_sequence import AnimationObject, make_concurrent
from fm.enums import Features, Level3, RightOfWay, TurnDirection, Zones, Specials, Level2
from interpolation import interpolate_single, InterpolatedLine
from util import TrackProperties, get_track_points, get_track_points_from_center, track_properties_to_colors, choose_closest_track_point, get_intersect, \
    get_continued_interpolation_animation, TrackPoint, get_continued_interpolation_line, zones_to_color, ZoneTypes, ZoneMisc
from manim import *


class Feature:
    """
    A FeatureIDE feature
    """

    def __init__(self, name, sub_features=[], mandatory=None, alternative=None, suffix=None):
        self.name = name
        self.mandatory = mandatory
        self.sub_features = sub_features
        self.alternative = alternative
        self.value = False
        self.type = name
        if suffix:
            self.apply_suffix(suffix)

    def __str__(self):
        if len(self.sub_features) > 0:
            selection = 'xor' if self.alternative else 'or'
            subs = str([str(sub_feature) for sub_feature in self.sub_features])
            return '{}: {}{}'.format(self.name, selection, subs)
        else:
            if self.value:
                return colored(self.name, 'green')
            else:
                return self.name

    def __bool__(self):
        return self.value

    def apply_suffix(self, suffix):
        self.name = "{} {}".format(self.name, suffix)
        # print(self.name)
        for sub_feature in self.sub_features:
            sub_feature.apply_suffix(suffix)

    def get_xml(self):
        if len(self.sub_features) > 0:
            selection = "alt" if self.alternative else "and"
            if self.mandatory:
                xml = ET.Element(selection, abstract="true", mandatory="true", name=self.name)
            else:
                xml = ET.Element(selection, abstract="true", name=self.name)
            for sub_feature in self.sub_features:
                xml.append(sub_feature.get_xml())
        else:
            xml = ET.Element("feature", name=self.name)
        return xml

    def get_mapping(self):
        key_value_pairs = [(self.name, self)]
        for sub_feature in self.sub_features:
            key_value_pairs += sub_feature.get_mapping()
        return key_value_pairs

    def get_selected_sub_features(self):
        selections = dict()
        selected_features = []
        for sub_feature in self.sub_features:
            if len(sub_feature.sub_features) > 0:
                selections[sub_feature.type] = sub_feature.get_selected_sub_features()
            elif sub_feature:
                selected_features.append(sub_feature.type)

        if len(selected_features) > 0:
            if len(selections.keys()) > 0:
                selected_features.append(selections)
            return selected_features
        else:
            return selections

    def get_num_possibilities(self):
        # print(colored("{}.get_num_possibilities():".format(self.name), 'blue'))
        sub_possibilites = []
        leaf_counter = 0
        for sub_feature in self.sub_features:
            if len(sub_feature.sub_features) > 0:
                sub_possibilites.append(sub_feature.get_num_possibilities())
            else:
                leaf_counter += 1

        # print("sub possbilites: {}, leafs: {}".format(sub_possibilites, leaf_counter))
        if self.alternative:
            num_possibilites = sum(sub_possibilites) + leaf_counter
            # print(colored("{} num possibilities: {}".format(self.name, num_possibilites), 'yellow'))
        elif not self.alternative:
            num_possibilites = math.prod(sub_possibilites) * (2 ** leaf_counter)
            # print(colored("{} num possibilities: {}".format(self.name, num_possibilites), 'cyan'))
        else:
            raise RuntimeError("Leaf Features should never be called!")

        return num_possibilites


class TLFeature(Feature):
    """
    Top Level Feature. A TLFeature represents a track element and can be drawn.
    """
    def __init__(self, name, track_property, suffix=None):
        super().__init__(name, mandatory=True, alternative=False, suffix=suffix)
        self.track_property = track_property
        self.zones = []
        self.start_or_end = None  # (['start', 'end'], zone_type)
        self.predecessor = []
        self.successor = []

    def get_features(self):
        raise NotImplementedError()

    def draw(self, track_width, color_by=None):
        """
        color_by in ['property', 'zone', None]
        """
        raise NotImplementedError()

    def get_collision_lines(self, track_width):
        raise NotImplementedError()

    def add_predecessor(self, element):
        self.predecessor.append(element)

    def add_successor(self, element):
        self.successor.append(element)

    def in_zone(self, zone_type):
        return zone_type in self.zones

    def is_zone_start(self):
        if self.start_or_end is None:
            return False, None
        else:
            start_or_end, zone_type = self.start_or_end
            if start_or_end == ZoneMisc.start:
                return True, zone_type
            else:
                return False, zone_type

    def is_zone_end(self):
        if self.start_or_end is None:
            return False, None
        else:
            start_or_end, zone_type = self.start_or_end
            if start_or_end == ZoneMisc.end:
                return True, zone_type
            else:
                return False, zone_type

    def get_color(self, color_by):
        if color_by == 'track_property':
            track_color = track_properties_to_colors([self.track_property])
        elif color_by == 'zone':
            track_color = zones_to_color(self.zones)
        elif color_by is None:
            track_color = WHITE
        else:
            track_color = color_by

        return track_color


class BasicFeature(TLFeature):
    """
    A basic feature, representing a single track element (e.g. 90 turn, single straight).
    Geometrically, the element is represented by center trackpoint of the start and end of the element
    start: Trackpoint(coords, direction)
    end:   Trackpoint(coords, direction)
    """

    def __init__(self, name, track_property, coords, start=None, end=None, suffix=None):
        super().__init__(name, track_property, suffix=suffix)
        self.coords = coords
        self.start = start
        self.end = end

    def scale(self, factor):
        self.start = TrackPoint(np.array(self.start.coords) * factor, self.start.direction)
        self.end = TrackPoint(np.array(self.end.coords) * factor, self.end.direction)

    def get_features(self):
        level3 = Feature(Features.level3.value, sub_features=[Feature(entry.value) for entry in Level3], mandatory=False, alternative=False)
        return [level3]

    def draw(self, track_width, z_index=0, color_by=None):
        track_color = self.get_color(color_by)

        right1, left1, center1 = get_track_points_from_center(self.start, track_width)
        right2, left2, center2 = get_track_points_from_center(self.end, track_width)
        px, py = interpolate_single(left1, left2)
        left_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
        px, py = interpolate_single(right1, right2)
        right_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
        px, py = interpolate_single(center1, center2)
        center_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)

        if not self.in_zone(ZoneTypes.no_passing):
            # distance = np.linalg.norm(np.array(center1.coords) - np.array(center2.coords))
            # center_line = DashedVMobject(center_line, num_dashes=int(6 * distance), positive_space_ratio=0.6)
            center_line = DashedVMobject(center_line, num_dashes=5, positive_space_ratio=0.6)

        return AnimationObject(type='play', content=[Create(right_line), Create(left_line), Create(center_line)], duration=0.25, bring_to_front=True, z_index=z_index)

    def get_collision_lines(self, track_width):
        right1, left1, center1 = get_track_points_from_center(self.start, track_width)
        right2, left2, center2 = get_track_points_from_center(self.end, track_width)
        left_line = InterpolatedLine(*interpolate_single(left1, left2))
        right_line = InterpolatedLine(*interpolate_single(right1, right2))
        # if self.sub_features['center_line'] == 'solid':
        #     center_line = InterpolatedLine(*interpolate_single(center1, center2))
        #     return [left_line, right_line, center_line]

        return [left_line, right_line]


class StraightStreet(BasicFeature):
    """
    A feature representing a single straight street
    """

    def get_features(self):
        """
        this depends on the zone now!
        """
        if self.in_zone(ZoneTypes.urban_area):
            self.alternative = True
            level2 = [Feature(entry.value) for entry in Level2]
            # level3 = super().get_features()
            return level2  # + level3
        elif self.in_zone(ZoneTypes.express_way) or self.in_zone(ZoneTypes.no_passing):
            # No obstacles on motorway!
            return [Feature(Features.level3.value, sub_features=[Feature(entry) for entry in [Level3.missing_left.value, Level3.missing_right.value]], mandatory=False, alternative=False)]
        elif self.in_zone(ZoneTypes.parking):
            return [Feature('parking')]
        else:
            return super().get_features()


class CurvedStreet(BasicFeature):
    """
    A feature representing a single curved street
    """

    def get_features(self):
        return super().get_features()


class IntersectionConnector(CurvedStreet):
    """
    A feature representing a single curved street
    """
    def __init__(self, name, track_property, coords, exiting, entering, start=None, end=None, suffix=None):
        super().__init__(name, track_property, coords, start=start, end=end, suffix=suffix)
        self.exiting = exiting
        self.entering = entering


class CompositeFeature(TLFeature):
    """
    A composite feature, representing a composite of multiple track elements (e.g. intersection, straight)
    """

    def __init__(self, name, track_property, coords_list, suffix=None):
        super().__init__(name, track_property, suffix=suffix)
        self.coords_list = coords_list

    def get_features(self):
        raise NotImplementedError()

    def draw(self, track_width):
        raise NotImplementedError()


class Intersection(CompositeFeature):
    """
    A feature representing an intersection connecting to 4 track pieces
    Geometrically, an intersection is represented by the left and right points of each entry
    bottom_left: Trackpoint(coords, direction)
    bottom_right: Trackpoint(coords, direction)
    top_left: Trackpoint(coords, direction)
    top_right: Trackpoint(coords, direction)
    """
    def __init__(self, name, coords_list, bottom_left=None, bottom_right=None, top_left=None, top_right=None, suffix=None):
        super().__init__(name, track_property=TrackProperties.intersection, coords_list=coords_list, suffix=suffix)
        self.center = np.array([coords_list]).mean(axis=1)
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_left = top_left
        self.top_right = top_right

    def scale(self, factor):
        self.bottom_left = TrackPoint(np.array(self.bottom_left.coords) * factor, self.bottom_left.direction)
        self.bottom_right = TrackPoint(np.array(self.bottom_right.coords) * factor, self.bottom_right.direction)
        self.top_left = TrackPoint(np.array(self.top_left.coords) * factor, self.top_left.direction)
        self.top_right = TrackPoint(np.array(self.top_right.coords) * factor, self.top_right.direction)

    def set_bottom_left(self, track_point1, track_point2):
        self.bottom_left = choose_closest_track_point(self.center, [track_point1, track_point2])
        return self

    def set_bottom_right(self, track_point1, track_point2):
        self.bottom_right = choose_closest_track_point(self.center, [track_point1, track_point2])
        return self

    def set_top_left(self, track_point1, track_point2):
        self.top_left = choose_closest_track_point(self.center, [track_point1, track_point2])
        return self

    def set_top_right(self, track_point1, track_point2):
        self.top_right = choose_closest_track_point(self.center, [track_point1, track_point2])
        return self

    def get_features(self):
        right_of_way = Feature(Features.right_of_way.value, sub_features=[Feature(entry.value) for entry in RightOfWay], mandatory=True, alternative=True)
        turn_direction = Feature(Features.turn_direction.value, sub_features=[Feature(entry.value) for entry in TurnDirection], mandatory=True, alternative=True)
        return [right_of_way, turn_direction]

    def draw(self, track_width, z_index=0, color_by=None):
        track_color = self.get_color(color_by)

        if self.bottom_left is None or self.bottom_right is None or self.top_left is None or self.top_right is None:
            raise ValueError('Geometric description not complete. At least on track point is missing')

        #### Fixate directions ####
        # bottom_left points to top_right
        # bottom_left and top_right point in the same direction
        direction = np.array(self.top_right.coords) - np.array(self.bottom_left.coords)
        direction = direction / np.linalg.norm(direction)
        if len(direction) > 2:
            direction = np.array(direction[:2])
        self.top_right.direction = direction
        self.bottom_left.direction = direction
        # top_left points to bottom_right
        # top_left and bottom_right point in the same direction
        direction = np.array(self.bottom_right.coords) - np.array(self.top_left.coords)
        direction = direction / np.linalg.norm(direction)
        if len(direction) > 2:
            direction = np.array(direction[:2])
        self.bottom_right.direction = direction
        self.top_left.direction = direction

        #### Find points of Intersections ####
        right1, left1, center1 = get_track_points_from_center(self.top_right, track_width)  # Think of this as bottom left
        right2, left2, center2 = get_track_points_from_center(self.bottom_left, track_width)  # Think of this as top right
        right3, left3, center3 = get_track_points_from_center(self.bottom_right, track_width)  # Think of this as top left
        right4, left4, center4 = get_track_points_from_center(self.top_left, track_width)  # Think of this as bottom right
        left_corner = get_intersect(left1.coords, left2.coords, right3.coords, right4.coords)
        bottom_corner = get_intersect(right1.coords, right2.coords, right3.coords, right4.coords)
        top_corner = get_intersect(left1.coords, left2.coords, left3.coords, left4.coords)
        right_corner = get_intersect(right1.coords, right2.coords, left3.coords, left4.coords)
        left_bottom_center = np.array([left_corner, bottom_corner]).mean(axis=0)
        bottom_right_center = np.array([bottom_corner, right_corner]).mean(axis=0)
        right_top_center = np.array([right_corner, top_corner]).mean(axis=0)
        top_left_center = np.array([top_corner, left_corner]).mean(axis=0)

        #### Draw all the lines ####
        anim_sequence = [
            # bottom left track
            # get_continued_interpolation(right1, top_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(right1, bottom_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(left1, left_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(center1, left_bottom_center, dashed=True, track_color=track_color, z_index=z_index),
            # top right track
            get_continued_interpolation_animation(right2, right_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(left2, top_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(center2, right_top_center, dashed=True, track_color=track_color, z_index=z_index),
            # top left track
            get_continued_interpolation_animation(right3, left_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(left3, top_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(center3, top_left_center, dashed=True, track_color=track_color, z_index=z_index),
            # bottom right track
            get_continued_interpolation_animation(right4, bottom_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(left4, right_corner, dashed=False, track_color=track_color, z_index=z_index),
            get_continued_interpolation_animation(center4, bottom_right_center, dashed=True, track_color=track_color, z_index=z_index)
        ]

        # make concurrent
        anim_sequence = [[anim_object] for anim_object in anim_sequence]
        return make_concurrent(anim_sequence)[0]

    def get_collision_lines(self, track_width):
        if self.bottom_left is None or self.bottom_right is None or self.top_left is None or self.top_right is None:
            raise ValueError('Geometric description not complete. At least on track point is missing')

        #### Fixate directions ####
        # bottom_left points to top_right
        # bottom_left and top_right point in the same direction
        direction = np.array(self.top_right.coords) - np.array(self.bottom_left.coords)
        direction = direction / np.linalg.norm(direction)
        if len(direction) > 2:
            direction = np.array(direction[:2])
        self.top_right.direction = direction
        self.bottom_left.direction = direction
        # top_left points to bottom_right
        # top_left and bottom_right point in the same direction
        direction = np.array(self.bottom_right.coords) - np.array(self.top_left.coords)
        direction = direction / np.linalg.norm(direction)
        if len(direction) > 2:
            direction = np.array(direction[:2])
        self.bottom_right.direction = direction
        self.top_left.direction = direction

        #### Find points of Intersections ####
        right1, left1, center1 = get_track_points_from_center(self.top_right, track_width)  # Think of this as bottom left
        right2, left2, center2 = get_track_points_from_center(self.bottom_left, track_width)  # Think of this as top right
        right3, left3, center3 = get_track_points_from_center(self.bottom_right, track_width)  # Think of this as top left
        right4, left4, center4 = get_track_points_from_center(self.top_left, track_width)  # Think of this as bottom right
        left_corner = get_intersect(left1.coords, left2.coords, right3.coords, right4.coords)
        bottom_corner = get_intersect(right1.coords, right2.coords, right3.coords, right4.coords)
        top_corner = get_intersect(left1.coords, left2.coords, left3.coords, left4.coords)
        right_corner = get_intersect(right1.coords, right2.coords, left3.coords, left4.coords)
        left_bottom_center = np.array([left_corner, bottom_corner]).mean(axis=0)
        bottom_right_center = np.array([bottom_corner, right_corner]).mean(axis=0)
        right_top_center = np.array([right_corner, top_corner]).mean(axis=0)
        top_left_center = np.array([top_corner, left_corner]).mean(axis=0)

        #### Draw all the lines ####
        lines = [
            # bottom left track
            get_continued_interpolation_line(right1, bottom_corner),
            get_continued_interpolation_line(left1, left_corner),
            # top right track
            get_continued_interpolation_line(right2, right_corner),
            get_continued_interpolation_line(left2, top_corner),
            # top left track
            get_continued_interpolation_line(right3, left_corner),
            get_continued_interpolation_line(left3, top_corner),
            # bottom right track
            get_continued_interpolation_line(right4, bottom_corner),
            get_continued_interpolation_line(left4, right_corner),
        ]
        # TODO: make these arguments or based on feature selection!
        no_passing_diagonal1 = False  # bottom_left to top_right
        no_passing_diagonal2 = False  # top_left to bottom_right

        if no_passing_diagonal1:
            lines += [
                # bottom left track
                get_continued_interpolation_line(center1, left_bottom_center),
                # top right track
                get_continued_interpolation_line(center2, right_top_center),
            ]
        if no_passing_diagonal2:
            lines += [
                # top left track
                get_continued_interpolation_line(center3, top_left_center),
                # bottom right track
                get_continued_interpolation_line(center4, bottom_right_center)
            ]

        return lines


# class Straight(CompositeFeature):
#     """
#     A feature representing a straight of fixed length
#     Geometrically, the element is represented by the left and right points of start and end
#     start: Trackpoint(coords, direction)
#     end: Trackpoint(coords, direction)
#     """
#     def __init__(self, name, coords_list, start=None, end=None, suffix=None):
#         super().__init__(name, track_property=TrackProperties.straight, coords_list=coords_list, suffix=suffix)
#         self.length = len(coords_list)
#         self.start = start
#         self.end = end
#
#     def scale(self, factor):
#         self.start = TrackPoint(np.array(self.start.coords) * factor, self.start.direction)
#         self.end = TrackPoint(np.array(self.end.coords) * factor, self.end.direction)
#
#     def get_features(self):
#         motorway = Feature(Features.zone.value, sub_features=[Feature(entry.value) for entry in Zones], mandatory=True, alternative=True)
#         special_subs = [Feature(entry.value) for entry in Specials] + [Feature('parking', sub_features=[Feature('left'), Feature('right')], alternative=False)]
#         special_element = Feature(Features.special.value, sub_features=special_subs, mandatory=True, alternative=True)
#         return [motorway, special_element]
#
#     def draw(self, track_width, z_index=0, color_by=None):
#         track_color = self.get_color(color_by)
#
#         right1, left1, center1 = get_track_points_from_center(self.start, track_width)
#         right2, left2, center2 = get_track_points_from_center(self.end, track_width)
#         px, py = interpolate_single(left1, left2)
#         left_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
#         px, py = interpolate_single(right1, right2)
#         right_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
#         px, py = interpolate_single(center1, center2)
#         center_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
#         distance = np.linalg.norm(np.array(center1.coords) - np.array(center2.coords))
#         # center_line = DashedVMobject(center_line, num_dashes=int(6 * distance), positive_space_ratio=0.6)
#         center_line = DashedVMobject(center_line, num_dashes=5 * self.length, positive_space_ratio=0.6)
#         return AnimationObject(type='play', content=[Create(right_line), Create(left_line), Create(center_line)], duration=0.25, bring_to_front=True, z_index=z_index)
#
#     def get_collision_lines(self, track_width):
#         right1, left1, center1 = get_track_points_from_center(self.start, track_width)
#         right2, left2, center2 = get_track_points_from_center(self.end, track_width)
#         left_line = InterpolatedLine(*interpolate_single(left1, left2))
#         right_line = InterpolatedLine(*interpolate_single(right1, right2))
#         # if self.sub_features['center_line'] == 'solid':
#         #     center_line = InterpolatedLine(*interpolate_single(center1, center2))
#         #     return [left_line, right_line, center_line]
#
#         return [left_line, right_line]


# if __name__ == '__main__':
    # feature = Straight('intersection', coords_list=None, start=None, end=None)
    # feature.sub_features[0].sub_features[0].value = True
    # feature.sub_features[1].sub_features[1].value = True
    # feature.sub_features[1].sub_features[3].sub_features[0].value = True
    # feature.sub_features[1].sub_features[3].sub_features[1].value = True
    # print(feature.get_selected_sub_features())
