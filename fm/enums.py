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
    level2 = 'level2'
    level3 = 'level3'
    right_of_way = 'right of way'
    turn_direction = 'turn direction'
    zone = 'zone'
    special = 'special'


class Level3(Enum):
    missing_right = 'missing_right'
    missing_left = 'missing_left'
    obstacle = 'obstacle'


class Level2(Enum):
    zebra = 'zebra'
    island = 'island'
    barred_area = 'barred_area'


class Zebra(Enum):
    island = 'zebra_island'
    pedestrian = 'pedestrian'


class Pedestrian(Enum):
    left = 'left'
    right = 'right'


class Offest(Enum):
    offset1 = 'offset1'
    offset2 = 'offset2'
    offset3 = 'offset3'


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
    # parking = 'parking'
    ramp = 'ramp'
    start_area = 'start area'
    none = 'none'


if __name__ == '__main__':
    # x = [entry.value for entry in LineMarkings]
    # print(x)
    print(TLFeatures.turn.value)
