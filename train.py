# originally from https://github.com/RolandZhu/robopianist



from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from robopianist import music

from mujoco_utils import composer_utils
from stable_baselines3.common.monitor import Monitor
from sbx.wrappers.monitor import SafeMonitor
from dmcgym import DMCGYM
from dm_env_wrappers import CanonicalSpecWrapper, ConcatObservationWrapper
from robopianist.wrappers import PianoSoundVideoWrapper, MidiEvaluationWrapper, SafeEnvironmentWrapper
import shimmy

def make_env(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,
             log_dir='./logs',
             timestep: int = 1e6):
    """
    Utility function for multiprocessed env.
    :param song: the name of the song
    :param seed: the inital seed for RNG
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load(song),
            trim_silence=True,
            control_timestep=0.05,
            gravity_compensation=True,
            primitive_fingertip_collisions=False,
            reduced_action_space=False,
            n_steps_lookahead=10,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            attachment_yaw=0.0,
        )

        env = composer_utils.Environment(
            task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
        )

        env = ConcatObservationWrapper(env)
        env = CanonicalSpecWrapper(env)
        if sound:
            env = PianoSoundVideoWrapper(
            env,
            record_every = 1,
            camera_id = None, # "piano/back",
            record_dir = f"./videos/{song}",
            song = song,
            timestep = timestep
            )
        env = MidiEvaluationWrapper(env)
        env = DMCGYM(env)
        env = shimmy.GymV21CompatibilityV0(env=env)
        env = Monitor(env, filename=f'results/{song}_{timestep}', info_keywords=('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1'))

        return env

    # set_random_seed(seed)

    return _init


def make_env_test(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,
             log_dir='./logs',
             timestep: int = 1e6):
    """
    Utility function for multiprocessed env.
    :param song: the name of the song
    :param seed: the inital seed for RNG
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load(song),
            trim_silence=True,
            control_timestep=0.05,
            gravity_compensation=True,
            primitive_fingertip_collisions=False,
            reduced_action_space=False,
            n_steps_lookahead=10,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            attachment_yaw=0.0,
        )

        env = composer_utils.Environment(
            task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
        )

        env = ConcatObservationWrapper(env)
        env = CanonicalSpecWrapper(env)
        if sound:
            env = PianoSoundVideoWrapper(
            env,
            record_every = 1,
            camera_id = None, # "piano/back",
            record_dir = f"./videos/{song}",
            song = song,
            timestep = timestep
            )
        env = MidiEvaluationWrapper(env)
        env = DMCGYM(env)
        env = shimmy.GymV21CompatibilityV0(env=env)
        env = Monitor(env, filename=f'results/{song}_{timestep}_test', info_keywords=('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1'))

        return env

    # set_random_seed(seed)

    return _init

def make_env_cl(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,
             log_dir='./logs',
             timestep: int = 6e5):
    """
    Utility function for multiprocessed env.
    :param song: the name of the song
    :param seed: the inital seed for RNG
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load(song),
            trim_silence=True,
            control_timestep=0.05,
            gravity_compensation=True,
            primitive_fingertip_collisions=False,
            reduced_action_space=False,
            n_steps_lookahead=10,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            attachment_yaw=0.0,
        )

        env = composer_utils.Environment(
            task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
        )

        env = ConcatObservationWrapper(env)
        env = CanonicalSpecWrapper(env)
        if sound:
            env = PianoSoundVideoWrapper(
            env,
            record_every = 1,
            camera_id = None, # "piano/back",
            record_dir = f"./videos/cl/{song}",
            song = song,
            timestep = timestep
            )
        env = MidiEvaluationWrapper(env)
        env = DMCGYM(env)
        env = shimmy.GymV21CompatibilityV0(env=env)
        env = Monitor(env, filename=f'results/cl/{song}_{timestep}', info_keywords=('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1'))

        return env

    # set_random_seed(seed)

    return _init


def make_env_cl_test(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,
             log_dir='./logs',
             timestep: int = 6e5):
    """
    Utility function for multiprocessed env.
    :param song: the name of the song
    :param seed: the inital seed for RNG
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load(song),
            trim_silence=True,
            control_timestep=0.05,
            gravity_compensation=True,
            primitive_fingertip_collisions=False,
            reduced_action_space=False,
            n_steps_lookahead=10,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            attachment_yaw=0.0,
        )

        env = composer_utils.Environment(
            task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
        )

        env = ConcatObservationWrapper(env)
        env = CanonicalSpecWrapper(env)
        if sound:
            env = PianoSoundVideoWrapper(
            env,
            record_every = 1,
            camera_id = None, # "piano/back",
            record_dir = f"./videos/cl/{song}",
            song = song,
            timestep = timestep
            )
        env = MidiEvaluationWrapper(env)
        env = DMCGYM(env)
        env = shimmy.GymV21CompatibilityV0(env=env)
        env = Monitor(env, filename=f'results/cl/{song}_{timestep}_test', info_keywords=('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1'))

        return env

    # set_random_seed(seed)

    return _init
