from manim import *


class AnimationSequence:
    def __init__(self):
        self.animations = list()

    def add_object(self, animation_object):
        self.animations.append(animation_object)


class AnimationObject:
    def __init__(self, type, content, wait_after=0, duration=0, bring_to_back=False, bring_to_front=False, z_index=None):
        self.type = type
        self.content = content
        self.duration = duration
        self.wait_after = wait_after
        self.bring_to_back = bring_to_back
        self.bring_to_front = bring_to_front
        self.z_index = z_index
        if not isinstance(content, list):
            self.content = [content]


class AnimationSequenceScene(MovingCameraScene):
    def construct(self):
        pass

    def play_animations(self, sequence):
        for animation in sequence:
            self.play_animation(animation)

    def play_concurrent(self, sequence_list):
        concurrent_sequence = make_concurrent(sequence_list)
        self.play_animations(concurrent_sequence)

    def play_animation(self, animation):
        if animation.bring_to_back or animation.bring_to_front or animation.z_index is not None:
            content = animation.content
            if animation.type == 'play':
                content = [c.mobject for c in animation.content]
            elif animation.bring_to_front:
                self.bring_to_front(*content)
            elif animation.bring_to_back:
                self.bring_to_back(*content)
            if animation.z_index is not None:
                [c.set_z_index(animation.z_index) for c in content]
        if animation.type == 'add':
            self.add(*animation.content)
            if animation.bring_to_front:
                self.bring_to_front(*content)
            if animation.bring_to_back:
                self.bring_to_back(*content)
        if animation.type == 'remove':
            self.remove(*animation.content)
        elif animation.type == 'play':
            self.play(*animation.content, run_time=animation.duration)

        self.wait(animation.wait_after)

    def move_camera(self, camera_size, camera_position, duration=1, border_scale=1.1, shift=[0, 0], resolution=[16, 9]):
        camera_position = [camera_position[0] + shift[0], camera_position[1] + shift[1]]
        self.play(
            self.camera.frame.animate.move_to((camera_position[0], camera_position[1], 0)),
            run_time=duration/2
        )
        if camera_size[0] / resolution[0] > camera_size[1] / resolution[1]:
            self.play(
                self.camera.frame.animate.set_width(camera_size[0] * border_scale),
                run_time=duration/2
            )
        else:
            self.play(
                self.camera.frame.animate.set_height(camera_size[1] * border_scale),
                run_time=duration/2
            )


def make_concurrent(sequence_list):
    """
    It is assumed that all sequences in the list are of the same length and have the same types at each time-step
    """
    concurrent_sequence = []
    first_sequence = sequence_list[0]
    time_steps = len(first_sequence)
    for time_step in range(time_steps):
        reference = first_sequence[time_step]
        animation_type = reference.type
        duration = reference.duration
        wait_after = reference.wait_after
        bring_to_back = reference.bring_to_back
        bring_to_front = reference.bring_to_front
        content = []
        for sequence in sequence_list:
            animation_object = sequence[time_step]
            if animation_type != animation_object.type:
                raise ValueError("Sequences type don't match at timestep {}. ({} vs {})".format(time_step, animation_type, animation_object.type))
            if duration < animation_object.duration:
                duration = animation_object.duration
            if wait_after < animation_object.wait_after:
                wait_after = animation_object.wait_after
            content += animation_object.content
        concurrent_sequence.append(AnimationObject(animation_type, content, wait_after, duration, bring_to_back, bring_to_front))
        return concurrent_sequence
