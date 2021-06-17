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