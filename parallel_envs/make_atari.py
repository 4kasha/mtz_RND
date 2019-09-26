from .atari_wrappers import make_atari, wrap_deepmind
from .monitor import Monitor
from .vec_env import SubprocVecEnv

from .env_eval import wrap_deepmind_eval
from .vec_env import Single_Env

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, max_episode_steps=4500):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id, max_episode_steps=max_episode_steps)
            env.seed(seed + rank)
            # for recording reward and episode length
            env = Monitor(env, filename=None, allow_early_resets=True)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_atari_env_for_eval(env_id, seed, wrapper_kwargs=None, max_episode_steps=4500):
    """
    Create a wrapped, monitored Single_Env for evalution.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env():
        def _thunk():
            env = make_atari(env_id, max_episode_steps=max_episode_steps)
            env.seed(seed)
            # for recording reward and episode length
            #env = Monitor(env, filename=None, allow_early_resets=True)
            return wrap_deepmind_eval(env, **wrapper_kwargs)
        return _thunk
    return Single_Env([make_env()])