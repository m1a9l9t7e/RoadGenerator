import json
from pprint import pprint


class Config:
    def __init__(self, path=None):
        if path is None:
            config_dict = dict()
        else:
            config_dict = json.load(open(path))

        layout = parse_layout(config_dict.get('layout'))
        zones = parse_zones(config_dict.get('zones'))
        features = parse_features(config_dict.get('features'))
        pprint(layout)


def parse_layout(layout_dict):
    if layout_dict is None:
        return dict()
    return layout_dict


def parse_zones(layout_dict):
    if layout_dict is None:
        return dict()
    return layout_dict


def parse_features(layout_dict):
    if layout_dict is None:
        return dict()
    return layout_dict


def generate_config():
    pass


if __name__ == '__main__':
    config = Config('/home/malte/PycharmProjects/circuit-creator/super_configs/config.json')
