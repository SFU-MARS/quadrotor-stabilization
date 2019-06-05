#!/usr/bin/python3.5

from pre_maml.envs.quad_falling_down import QuadFallingDown

from garage.envs.normalized_env import normalize
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.tf.algos.ppo import PPO
from garage.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy


env = normalize(QuadFallingDown())
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



