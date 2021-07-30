from enum import Enum, auto


class TLFeatures(Enum):
    """
    Top Level Features
    """
    default = 'single straight'
    turn = 'turn'
    intersection = 'intersection'
    straight = 'straight'


class Features(Enum):
    line_marking = 'line marking'
    right_of_way = 'right of way'
    turn_direction = 'turn direction'
    zone = 'zone'
    special = 'special'


class LineMarkings(Enum):
    dashed = 'dashed'
    solid = 'solid'
    double_solid = 'double solid'


class RightOfWay(Enum):
    go_ahead = 'go ahead'
    yield_ = 'yield'
    stop = 'stop'


class TurnDirection(Enum):
    straight = 'straight'
    left = 'left'
    right = 'right'


class Zones(Enum):
    urban_area = 'urban area'
    motorway = 'motorway'
    elevated = 'elevated'
    none = 'rural'


class Specials(Enum):
    parking = 'parking'
    ramp = 'ramp'
    start_area = 'start area'
    none = 'none'


if __name__ == '__main__':
    # x = [entry.value for entry in LineMarkings]
    # print(x)
    print(TLFeatures.turn.value)
