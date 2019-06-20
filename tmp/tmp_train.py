#!/usr/bin/python3.5

from pre_maml.envs.quad_falling_down import QuadFallingDown

from rllab.envs.normalized_env import normalize
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.ppo import PPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


env = normalize(env=QuadFallingDown())
policy = GaussianMLPPolicy(
			env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = PPO(
		env=env,
		policy=policy,
		baseline=baseline,
)
algo.train()



