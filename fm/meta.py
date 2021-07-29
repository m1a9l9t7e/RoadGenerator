import xml.etree.ElementTree as ET


class Feature:
    """
    A FeatureIDE feature
    """

    def __init__(self, name, sub_features=[], mandatory=None, alternative=None):
        self.name = name
        self.mandatory = mandatory
        self.sub_features = sub_features
        self.alternative = alternative

    def __str__(self):
        if len(self.sub_features) > 0:
            selection = 'xor' if self.alternative else 'or'
            subs = str([str(sub_feature) for sub_feature in self.sub_features])
            return '{}: {}{}'.format(self.name, selection, subs)
        return self.name

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


class BasicStreet(Feature):
    """
    A basic abstract street feature
    """

    def __init__(self, name):
        super().__init__(name, sub_features=self.get_features(), mandatory=True, alternative=False)

    def get_features(self):
        lane_markings = Feature('center line markings', sub_features=[Feature('dashed'), Feature('solid'), Feature('double solid')],
                                mandatory=True, alternative=True)
        return [lane_markings]


class StraightStreet(BasicStreet):
    """
    A feature representing a single straight street
    """

    def get_features(self):
        parent_features = super().get_features()
        features = []
        return parent_features + features


class CurvedStreet(BasicStreet):
    """
    A feature representing a single curved street
    """

    def get_features(self):
        parent_features = super().get_features()
        features = []
        return parent_features + features


class Intersection(Feature):
    """
    A basic abstract street feature
    """

    def __init__(self, name):
        super().__init__(name, sub_features=self.get_features(), mandatory=True, alternative=False)

    def get_features(self):
        right_of_way = Feature('right of way', sub_features=[Feature('yield'), Feature('stop'), Feature('go ahead')],
                               mandatory=True, alternative=True)
        turn_direction = Feature('turn direction', sub_features=[Feature('straight'), Feature('left'), Feature('right')],
                                 mandatory=True, alternative=True)
        return [right_of_way, turn_direction]


if __name__ == '__main__':
    feature = Intersection('intersection1')
    tree = ET.ElementTree(feature.get_xml())
    tree.write("filename.xml")
