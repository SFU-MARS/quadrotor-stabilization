from pre_maml.envs import QuadFallingDown
from garage.envs import normalize
from garage.np.baselines import LinearFeatureBaseline
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



