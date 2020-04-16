import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register envs
# ----------------------------------------
# Quadrotor Hovering from Ground

register(
        # For gym id, the correct form would be xxx-v0, not xxx_v0
        id='QuadTakeOffHoverEnv-v0',
        entry_point='gym_foo.gym_foo.envs:QuadTakeOffHoverEnv_v0',
)
