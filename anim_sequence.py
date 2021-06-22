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
