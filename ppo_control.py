import gym
import gym_minigrid
import stable_baselines
from gym_minigrid.wrappers import FullyObsOneHotWrapper, OneHotPartialObsWrapper, ImgObsWrapper
from stable_baselines import PPO2

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('MiniGrid-HallwayWithVictims-v0')
env = ImgObsWrapper(env)
env = FullyObsOneHotWrapper(env)
env = DummyVecEnv([lambda: env])
obs = env.reset()
print(obs)

# import ipdb;ipdb.set_trace()
model = PPO2(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=10000)

save_dir = './'
model.save(save_dir + "PPO2_tutorial")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
# 