import numpy as np
from manim import *
from anim_sequence import AnimationObject, AnimationSequenceScene, make_concurrent
from ip.iterative_construction import create_base_grid, Node, get_n, count_adjacent, get_combinations, make_next_grid
from util import GridShowCase, get_square, get_text, TrackProperties, TreeShowCase, tree_to_list, get_line
from tqdm import tqdm
from termcolor import colored


class DynamicIPVisualization:
    def __init__(self, width, height, show_text=True, scale=1, shift=[0, 0]):
        self.width, self.height = (width + 1, height + 1)
        self.ip_width, self.ip_height = (width, height)
        self.show_text = show_text
        self.n = np.ceil(self.ip_width / 2) * self.ip_height + np.floor(self.ip_width / 2)
        self.square_size = scale
        self.shift = shift
        self.num_elements = self.ip_width * self.ip_height
        self.helper = GridShowCase(self.num_elements, [self.square_size, self.square_size], spacing=[0, 0], space_ratio=[1, 1])
        self.animation_sequence = []

        # Descriptors
        self.negative_cell_desc = (BLACK, DARK_GREY, 'Negative')
        self.positive_cell_desc = (GREEN_E, DARK_GREY, 'Positive')
        self.blocked_cell_desc = (DARK_BROWN, DARK_GREY, 'Blocked')

        # Pre-declarations
        self.captions = []
        self.legend = []
        self.current_depth = -1
        self.depth_label = None

        self.grid, self.squares = self.init()

    def get_animation_sequence(self):
        self.add_pause(3)
        return self.animation_sequence

    def add_pause(self, duration):
        self.animation_sequence += [AnimationObject('wait', content=[], wait_after=duration)]

    def init(self, show=True):
        primary_color, secondary_color, _ = self.negative_cell_desc
        grid = np.zeros((self.ip_width, self.ip_height))

        squares = []
        captions = []
        square_grid = [[None for y in range(self.ip_height)] for x in range(self.ip_width)]
        for x in range(self.ip_width):
            for y in range(self.ip_height):
                square = get_square(self.transform_coords((x, y)), self.square_size, primary_color, secondary_color, border_width=3 * self.square_size)
                squares.append(square)
                square_grid[x][y] = square
                if self.show_text:
                    captions.append(get_text(r'$c_{' + str(x+1) + ',' + str(y+1) + '}$', self.transform_coords((x, y)), scale=self.square_size))

        self.animation_sequence += [
            AnimationObject('add', content=squares, z_index=0),
        ]

        if self.show_text:
            self.animation_sequence += [
                AnimationObject('add', content=captions, z_index=10),
            ]

        # self.add_legend()
        self.captions = captions
        return grid, square_grid

    def update(self, grid, animate=False, depth=None, duration=0.5):
        animation_sequences = []

        # unpack primary and secondary colors
        pc1, sc1, _ = self.positive_cell_desc
        pc2, sc2, _ = self.negative_cell_desc
        pc3, sc3, _ = self.blocked_cell_desc

        indices = []
        for x in range(self.ip_width):
            for y in range(self.ip_height):
                indices.append((x, y))

        for index, (x, y) in enumerate(indices):
            if grid[x][y] == self.grid[x][y]:
                continue

            # coords = self.helper.get_element_coords(index)
            square = self.squares[x][y]
            new_square = None
            if grid[x][y] == -1:
                new_square = get_square(self.transform_coords((x, y)), self.square_size, pc3, sc3, border_width=3 * self.square_size)
            elif grid[x][y] == 0:
                new_square = get_square(self.transform_coords((x, y)), self.square_size, pc2, sc2, border_width=3 * self.square_size)
            elif grid[x][y] == 1:
                new_square = get_square(self.transform_coords((x, y)), self.square_size, pc1, sc1, border_width=3 * self.square_size)

            if animate:
                animation_sequences.append([
                    AnimationObject('play', content=ReplacementTransform(square, new_square), z_index=0, duration=duration)
                ])
            else:
                animation_sequences.append([
                    AnimationObject('remove', content=square, z_index=0),
                    AnimationObject('add', content=new_square, z_index=0)
                ])

        # position, size, shift = self.get_camera_settings()
        # label_pos = (position[0] + size[0] / 2, position[1] + size[1] + self.square_size)
        if depth is not None:
            label_pos = (1.2, 2.75)
            depth_label = get_text('iteration depth = {}'.format(depth), label_pos, scale=self.square_size * 0.7)
            self.animation_sequence += [
                AnimationObject('add', content=depth_label, z_index=10)
            ]
            self.animation_sequence += make_concurrent(animation_sequences)
            self.animation_sequence += [
                AnimationObject('remove', content=depth_label, z_index=10),
            ]
        else:
            self.animation_sequence += make_concurrent(animation_sequences)

        # self.add_pause(1)
        self.grid = grid

    def remove_captions(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.captions),
        ]

    def add_legend(self):
        legend_entries = [
            self.blocked_cell_desc,
            self.positive_cell_desc,
            self.negative_cell_desc
        ]

        _, camera_size, _ = self.get_camera_settings()
        self.legend = get_legend(legend_entries, shift=[camera_size[0] + self.square_size / 4, self.square_size / 2], scale=self.square_size * 0.5)
        self.animation_sequence += [
            AnimationObject('add', content=self.legend, bring_to_front=True)
        ]

    def remove_legend(self):
        self.animation_sequence += [
            AnimationObject('remove', content=self.legend)
        ]

    def get_camera_settings(self):
        camera_position, camera_size = self.helper.get_global_camera_settings()
        shift = [-self.square_size / 2, -self.square_size / 2]
        return camera_position, camera_size, shift

    def transform_coords(self, coords):
        x, y = coords
        x_shift, y_shift = self.shift
        return [x + x_shift, y + y_shift]


def get_legend(legend_list, shift, scale=1.0):
    drawables = []
    helper = GridShowCase(len(legend_list) * 2, [scale, scale], spacing=[scale, scale/2], space_ratio=[2, len(legend_list)], shift=shift)
    for index, (_color, _secondary_color, label) in enumerate(legend_list):
        element_coords = helper.get_element_coords(index * 2)
        text_coords = helper.get_element_coords(index * 2 + 1)
        drawables += [
            get_square(element_coords, scale, _color, _secondary_color, border_width=2 * scale),
            get_text(label, (text_coords[0] + len(label) * scale * 0.05, text_coords[1]), scale=scale)
        ]

    return drawables


class Iterator:
    def __init__(self, width, height, _print=False):
        self.width = width
        self.height = height
        self._print = _print
        self.variants = []
        self.iteration_counter = 0
        self.leaf_counter = 0

    def iterate(self, viz=None, depth_first=True):
        root = NodeWrapper(grid=create_base_grid(np.zeros((self.width, self.height), dtype=int), positive_cells=[(0, 0)], negative_cells=[]), num_positives=1, depth=0)
        queue = [root]

        while len(queue) > 0:
            if depth_first:
                next_node = queue.pop(len(queue) - 1)
            else:
                next_node = queue.pop(0)

            if viz is not None:
                viz.update(next_node.grid, animate=True, depth=next_node.depth)
            _next = next_node.get_next()

            add_to_queue = self.unpack_next(_next)
            queue += add_to_queue
            self.iteration_counter += 1

           # if self.iteration_counter > 5:
           #     return root

        if self._print:
            print("Number of checked leafs: {}".format(self.leaf_counter))
            print("Number of total iterations: {}".format(self.iteration_counter))
            print("Number found variants: {}".format(len(self.variants)))

        return root

    def unpack_next(self, _next):
        leaf, content = _next
        add_to_queue = []
        if leaf:
            if content is not None:
                self.variants.append(content)
            self.leaf_counter += 1
        else:
            add_to_queue = content
        return add_to_queue


class NodeWrapper(Node):
    def __init__(self, grid, num_positives, depth, parent=None):
        super().__init__(grid, num_positives)
        self.depth = depth
        self.parent = parent
        self.children = []
        self.leaf = False
        self.variant = False
        self.position = None

    def get_next(self):
        # leaf node
        if self.num_positives == get_n(len(self.grid), len(self.grid[0])):
            self.leaf = True
            self.variant = True
            return True, self.grid

        possibilites = []
        indices = np.argwhere(self.grid == 0)
        for (x, y) in indices:
            if count_adjacent(self.grid, (x, y)) == 1:
                possibilites.append((x, y))

        # also leaf node, but invald
        if len(possibilites) == 0:
            self.leaf = True
            return True, None

        _next = []
        combinations = get_combinations(len(possibilites))
        for combination in combinations:
            next_grid, counter, success = make_next_grid(self.grid, combination, np.array(possibilites))
            if not success:
                continue
            child = NodeWrapper(next_grid, self.num_positives + counter, depth=self.depth+1, parent=self)
            self.children.append(child)
            _next.append(child)

        return False, _next


class Iteration(AnimationSequenceScene):
    def construct(self):
        width, height = (4, 4)
        viz = DynamicIPVisualization(width-1, height-1)
        camera_position, camera_size, shift = viz.get_camera_settings()

        # Do Iteration
        iterator = Iterator(width-1, height-1)
        iterator.iterate(viz, depth_first=False)

        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.7, shift=shift)
        self.play_animations(viz.get_animation_sequence())
        self.wait(3)


class IterationTree(AnimationSequenceScene):
    def construct(self):
        zoomed_out = 6
        zoomed_in = 1.25
        time_at_node = 1
        width, height = (4, 4)
        iterator = Iterator(width-1, height-1)
        root = iterator.iterate(depth_first=False)

        reference_node = DynamicIPVisualization(width - 1, height - 1)
        _, element_size, _ = reference_node.get_camera_settings()
        tree_showcase = TreeShowCase(root, element_dimensions=element_size, spacing=element_size)
        elements = tree_to_list(root, dfs=False)
        print("Number of Nodes in Tree: {}".format(len(elements)))
        camera_position, camera_size = tree_showcase.get_zoomed_camera_settings(elements[0])
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.25, shift=-np.array(element_size) / 5)
        previous_element = None

        for index, element in tqdm(enumerate(elements)):
            viz = DynamicIPVisualization(width - 1, height - 1, shift=tree_showcase.get_element_coords(element), show_text=True)
            viz.update(element.grid, animate=False)
            camera_position, camera_size = tree_showcase.get_zoomed_camera_settings(element)
            if previous_element is None:
                self.play_animations(viz.get_animation_sequence())
                self.move_camera(camera_size, camera_position, duration=0.1, border_scale=zoomed_in, shift=-np.array(element_size) / 5)
            else:
                if element.parent == previous_element:
                    previous_position, _ = tree_showcase.get_zoomed_camera_settings(previous_element)
                    line = get_line(previous_position, camera_position, stroke_width=10)
                    # transition camera smoothly
                    self.move_camera(camera_size, previous_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    self.play_animation(AnimationObject('play', content=Create(line), duration=0.5, z_index=-5))
                    if element.variant:
                        glow = 0.1
                        back_drop = get_square(np.array(tree_showcase.get_element_coords(element)) + (0.225 + glow) * np.array(element_size), element_size[0] * (1 + glow), YELLOW_D, YELLOW_B, border_width=5)
                        element.export_grid()
                        viz.update(element.grid, animate=True, duration=1)
                        self.play_animations(viz.get_animation_sequence())
                        self.play_animation(AnimationObject('play', content=Create(back_drop), duration=1, z_index=-4))
                        break
                    else:
                        self.play_animations(viz.get_animation_sequence())
                    self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    # self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_in, shift=-np.array(element_size)/5)
                else:
                    previous_position, _ = tree_showcase.get_zoomed_camera_settings(previous_element)
                    parent_position, _ = tree_showcase.get_zoomed_camera_settings(element.parent)
                    line = get_line(parent_position, camera_position, stroke_width=10)
                    # transition camera smoothly
                    self.move_camera(camera_size, previous_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    self.move_camera(camera_size, parent_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    self.play_animation(AnimationObject('play', content=Create(line), duration=0.5, z_index=-5))
                    if element.variant:
                        glow = 0.1
                        back_drop = get_square(np.array(tree_showcase.get_element_coords(element)) + (0.225 + glow) * np.array(element_size), element_size[0] * (1 + glow), YELLOW_D, YELLOW_B, border_width=5)
                        element.export_grid()
                        viz.update(element.grid, animate=True, duration=1)
                        self.play_animations(viz.get_animation_sequence())
                        self.play_animation(AnimationObject('play', content=Create(back_drop), duration=1, z_index=-4))
                    else:
                        self.play_animations(viz.get_animation_sequence())
                    self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    # self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_in, shift=-np.array(element_size)/5)
            # if index > 15:
            #     break

            previous_element = element
            # self.wait(1)

        camera_position, camera_size = tree_showcase.get_global_camera_settings()
        self.move_camera(camera_size, camera_position, duration=4, border_scale=1)
        self.wait(6)


class CustomIterationTree(AnimationSequenceScene):
    def construct(self):
        zoomed_out = 6
        zoomed_in = 1.25
        time_at_node = 1
        width, height = (4, 4)
        iterator = Iterator(width-1, height-1)
        root = iterator.iterate(depth_first=False)

        reference_node = DynamicIPVisualization(width - 1, height - 1)
        _, element_size, _ = reference_node.get_camera_settings()
        element_width, element_height = element_size
        tree_showcase = TreeShowCase(root, element_dimensions=element_size, spacing=(element_width * 0.66, element_height * 0.33))
        elements = tree_to_list(root, dfs=False)
        print("Number of Nodes in Tree: {}".format(len(elements)))
        camera_position, camera_size = tree_showcase.get_zoomed_camera_settings(elements[0])
        self.move_camera(camera_size, camera_position, duration=0.1, border_scale=1.25, shift=-np.array(element_size) / 5)
        previous_element = None

        for index, element in tqdm(enumerate(elements)):
            viz = DynamicIPVisualization(width - 1, height - 1, shift=tree_showcase.get_element_coords(element), show_text=True)
            viz.update(element.grid, animate=False)
            camera_position, camera_size = tree_showcase.get_zoomed_camera_settings(element)
            if previous_element is None:
                self.play_animations(viz.get_animation_sequence())
                self.move_camera(camera_size, camera_position, duration=0.1, border_scale=zoomed_in, shift=-np.array(element_size) / 5)
            else:
                if element.parent == previous_element:
                    previous_position, _ = tree_showcase.get_zoomed_camera_settings(previous_element)
                    line = get_line(previous_position, camera_position, stroke_width=10)
                    # transition camera smoothly
                    self.move_camera(camera_size, previous_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    self.play_animation(AnimationObject('play', content=Create(line), duration=0.5, z_index=-5))
                    if element.variant:
                        glow = 0.1
                        back_drop = get_square(np.array(tree_showcase.get_element_coords(element)) + (0.225 + glow) * np.array(element_size), element_size[0] * (1 + glow), YELLOW_D, YELLOW_B, border_width=5)
                        element.export_grid()
                        viz.update(element.grid, animate=True, duration=1)
                        self.play_animations(viz.get_animation_sequence())
                        self.play_animation(AnimationObject('play', content=Create(back_drop), duration=1, z_index=-4))
                        break
                    else:
                        self.play_animations(viz.get_animation_sequence())
                    self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    # self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_in, shift=-np.array(element_size)/5)
                else:
                    previous_position, _ = tree_showcase.get_zoomed_camera_settings(previous_element)
                    parent_position, _ = tree_showcase.get_zoomed_camera_settings(element.parent)
                    line = get_line(parent_position, camera_position, stroke_width=10)
                    # transition camera smoothly
                    self.move_camera(camera_size, previous_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    self.move_camera(camera_size, parent_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    self.play_animation(AnimationObject('play', content=Create(line), duration=0.5, z_index=-5))
                    if element.variant:
                        glow = 0.1
                        back_drop = get_square(np.array(tree_showcase.get_element_coords(element)) + (0.225 + glow) * np.array(element_size), element_size[0] * (1 + glow), YELLOW_D, YELLOW_B, border_width=5)
                        element.export_grid()
                        viz.update(element.grid, animate=True, duration=1)
                        self.play_animations(viz.get_animation_sequence())
                        self.play_animation(AnimationObject('play', content=Create(back_drop), duration=1, z_index=-4))
                    else:
                        self.play_animations(viz.get_animation_sequence())
                    self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_out, shift=-np.array(element_size)/5)
                    # self.move_camera(camera_size, camera_position, duration=1, border_scale=zoomed_in, shift=-np.array(element_size)/5)
            if index > 3:
                break

            previous_element = element
            # self.wait(1)

        anim_sequence = []

        for index, label_position in enumerate(tree_showcase.get_label_positions(shift=[8, -0.4])):
            label = get_text('$\\bar{C}_' + str(index + 1) + '$', coords=label_position, scale=5)
            # label = get_text('Hello World!', coords=label_position, scale=5)
            anim_sequence.append(AnimationObject(type='add', content=label))

        self.play_animations(anim_sequence)

        camera_position, camera_size = tree_showcase.get_global_camera_settings()
        camera_width, camera_height = camera_size
        self.move_camera((camera_width * 0.9, camera_height * 0.5), camera_position, duration=4, border_scale=1)
        self.wait(6)


if __name__ == '__main__':
    scene = IterationTree()
    scene.construct()
