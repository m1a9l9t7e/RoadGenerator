import xml.etree.ElementTree as ET
from termcolor import colored
from fm.enums import Features, LineMarkings, RightOfWay, TurnDirection, Zones, Specials


class Feature:
    """
    A FeatureIDE feature
    """

    def __init__(self, name, sub_features=[], mandatory=None, alternative=None, suffix=None):
        self.name = name
        self.mandatory = mandatory
        self.sub_features = sub_features
        self.alternative = alternative
        self.value = None
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


class BasicFeature(Feature):
    """
    A basic feature, representing a single track element (e.g. 90 turn, single straight)
    """

    def __init__(self, name, coords, suffix=None):
        super().__init__(name, sub_features=self.get_features(), mandatory=True, alternative=False, suffix=suffix)
        self.coords = coords

    def get_features(self):
        lane_markings = Feature(Features.line_marking.value, sub_features=[Feature(entry.value) for entry in LineMarkings],
                                mandatory=True, alternative=True)
        return [lane_markings]


class CompositeFeature(Feature):
    """
    A composite feature, representing a composite of multiple track elements (e.g. intersection, straight)
    """

    def __init__(self, name, coords_list, suffix=None):
        super().__init__(name, sub_features=self.get_features(), mandatory=True, alternative=False, suffix=suffix)
        self.coords_list = coords_list

    def get_features(self):
        return []


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


class Intersection(CompositeFeature):
    """
    A feature representing an intersection consisting of 4 tiles
    """

    def get_features(self):
        right_of_way = Feature(Features.right_of_way.value, sub_features=[Feature(entry.value) for entry in RightOfWay], mandatory=True, alternative=True)
        turn_direction = Feature(Features.turn_direction.value, sub_features=[Feature(entry.value) for entry in TurnDirection], mandatory=True, alternative=True)
        return [right_of_way, turn_direction]


class Straight(CompositeFeature):
    """
    A feature representing a straight of fixed length
    """

    def get_features(self):
        motorway = Feature(Features.zone.value, sub_features=[Feature(entry.value) for entry in Zones], mandatory=True, alternative=True)
        special_element = Feature(Features.special.value, sub_features=[Feature(entry.value) for entry in Specials], mandatory=True, alternative=True)
        return [motorway, special_element]


if __name__ == '__main__':
    feature = Intersection('intersection', [(1, 1), (1, 2), (2, 1), (2, 2)])
    print(feature)
