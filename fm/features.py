import xml.etree.ElementTree as ET
from termcolor import colored
from anim_sequence import AnimationObject
from fm.enums import Features, LineMarkings, RightOfWay, TurnDirection, Zones, Specials
from interpolation import interpolate_single
from util import TrackProperties, get_track_points, get_track_points_from_center, track_properties_to_colors
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


class TLFeature(Feature):
    """
    Top Level Feature. A TLFeature represents a track element and can be drawn.
    """
    def __init__(self, name, track_property, suffix=None):
        super().__init__(name, sub_features=self.get_features(), mandatory=True, alternative=False, suffix=suffix)
        self.track_property = track_property

    def get_features(self):
        raise NotImplementedError()

    def draw(self, track_width):
        raise NotImplementedError()


class BasicFeature(TLFeature):
    """
    A basic feature, representing a single track element (e.g. 90 turn, single straight).
    Geometrically, the element is represented by the left and right points of start and end
    start: Trackpoint(coords, direction)
    end:   Trackpoint(coords, direction)
    """

    def __init__(self, name, track_property, coords, start=None, end=None, suffix=None):
        super().__init__(name, track_property, suffix=suffix)
        self.coords = coords
        self.start = start
        self.end = end

    def get_features(self):
        lane_markings = Feature(Features.line_marking.value, sub_features=[Feature(entry.value) for entry in LineMarkings],
                                mandatory=True, alternative=True)
        return [lane_markings]

    def draw(self, track_width, z_index=0):
        track_color = track_properties_to_colors([self.track_property])
        right1, left1, center1 = get_track_points_from_center(self.start, track_width)
        right2, left2, center2 = get_track_points_from_center(self.end, track_width)
        px, py = interpolate_single(left1, left2)
        left_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
        px, py = interpolate_single(right1, right2)
        right_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
        px, py = interpolate_single(center1, center2)
        center_line = ParametricFunction(function=lambda t: (px(t), py(t), 0), color=track_color, stroke_width=2)
        center_line = DashedVMobject(center_line, num_dashes=5, positive_space_ratio=0.6)
        return AnimationObject(type='play', content=[Create(right_line), Create(left_line), Create(center_line)], duration=0.25, bring_to_front=True, z_index=z_index)


class StraightStreet(BasicFeature):
    """
    A feature representing a single straight street
    """

    def get_features(self):
        parent_features = super().get_features()
        features = []
        return parent_features + features


class CurvedStreet(BasicFeature):
    """
    A feature representing a single curved street
    """

    def get_features(self):
        parent_features = super().get_features()
        features = []
        return parent_features + features


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
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_left = top_left
        self.top_right = top_right

    def get_features(self):
        right_of_way = Feature(Features.right_of_way.value, sub_features=[Feature(entry.value) for entry in RightOfWay], mandatory=True, alternative=True)
        turn_direction = Feature(Features.turn_direction.value, sub_features=[Feature(entry.value) for entry in TurnDirection], mandatory=True, alternative=True)
        return [right_of_way, turn_direction]

    def draw(self, track_width):
        # raise NotImplementedError()
        return None


class Straight(CompositeFeature):
    """
    A feature representing a straight of fixed length
    Geometrically, the element is represented by the left and right points of start and end
    start: Trackpoint(coords, direction)
    end: Trackpoint(coords, direction)
    """
    def __init__(self, name, coords_list, start=None, end=None, suffix=None):
        super().__init__(name, track_property=TrackProperties.straight, coords_list=coords_list, suffix=suffix)
        self.start = start
        self.end = end

    def get_features(self):
        motorway = Feature(Features.zone.value, sub_features=[Feature(entry.value) for entry in Zones], mandatory=True, alternative=True)
        special_subs = [Feature(entry.value) for entry in Specials] + [Feature('parking', sub_features=[Feature('left'), Feature('right')], alternative=False)]
        special_element = Feature(Features.special.value, sub_features=special_subs, mandatory=True, alternative=True)
        return [motorway, special_element]

    def draw(self, track_width):
        # raise NotImplementedError()
        return None


if __name__ == '__main__':
    feature = Straight('intersection', coords_list=None, start=None, end=None)
    feature.sub_features[0].sub_features[0].value = True
    feature.sub_features[1].sub_features[1].value = True
    feature.sub_features[1].sub_features[3].sub_features[0].value = True
    feature.sub_features[1].sub_features[3].sub_features[1].value = True
    print(feature.get_selected_sub_features())
