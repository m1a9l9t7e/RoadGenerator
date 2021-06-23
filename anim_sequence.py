from manim import *


class AnimationSequence:
    def __init__(self):
        self.animations = list()

    def add_object(self, animation_object):
        self.animations.append(animation_object)


class AnimationObject:
    def __init__(self, type, content, wait_after=0, duration=0, bring_to_back=False, bring_to_front=False):
        self.type = type
        self.content = content
        self.duration = duration
        self.wait_after = wait_after
        self.bring_to_back = bring_to_back
        self.bring_to_front = bring_to_front
        if not isinstance(content, list):
            self.content = [content]


class AnimationSequenceScene(MovingCameraScene):
    def construct(self):
        pass

    def play_animations(self, sequence):
        for animation in sequence:
            self.play_animation(animation)

    def play_concurrent(self, sequence_list):
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

        self.play_animations(concurrent_sequence)

    def play_animation(self, animation):
        if animation.bring_to_back or animation.bring_to_front:
            content = animation.content
            if animation.type == 'play':
                content = [c.mobject for c in animation.content]
            if animation.bring_to_front:
                self.bring_to_front(*content)
            if animation.bring_to_back:
                self.bring_to_back(*content)
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
