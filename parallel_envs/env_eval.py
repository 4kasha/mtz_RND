# overlay for evaluation

import numpy as np
import cv2

from .atari_wrappers import WarpFrame, FrameStack, LazyFrames

cv2.ocl.setUseOpenCL(False)

class WarpFrame_eval(WarpFrame):
    """Returns an original RGB frame and warpped frame to 84x84.
    Return shapes :
        _frame : (210, 160, 3)
        frame[None, None, :, :] : (1, 1, 84, 84)
    """
    def __init__(self, env):
        WarpFrame.__init__(self, env)

    def observation(self, frame):
        _frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return _frame, frame[None, None, :, :]

class FrameStack_eval(FrameStack):
    """Stack k last frames.
    Returns an original RGB frame and lazy array, which is much more memory efficient.
    Return shapes :
        _ob : (210, 160, 3)
        self._get_ob() : (1, k, 84, 84)
    """
    def __init__(self, env, k):
        FrameStack.__init__(self, env, k)

    def reset(self):
        _ob, ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return (_ob, self._get_ob())

    def step(self, action):
        (_ob, ob), reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return (_ob, self._get_ob()), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames_eval(list(self.frames))

class LazyFrames_eval(LazyFrames):
    """This object should only be converted to numpy array before being passed to the model."""
    def __init__(self, frames):
        LazyFrames.__init__(self, frames)

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=1)
            self._frames = None
        return self._out

def wrap_deepmind_eval(env, frame_stack=True):
    """Configure environment for DeepMind-style Atari for evalution."""
    env = WarpFrame_eval(env)
    if frame_stack:
        env = FrameStack_eval(env, 4)
    return env