import gym
import gym_minigrid
import stable_baselines
from gym_minigrid.wrappers import FullyObsOneHotWrapper, OneHotPartialObsWrapper, ImgObsWrapper, ViewSizeWrapper
from stable_baselines import PPO2

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('MiniGrid-Hall19x15seerescue2-v0')
env = ViewSizeWrapper(env, agent_view_size=7)
env = ImgObsWrapper(env)
env = FullyObsOneHotWrapper(env)
env = DummyVecEnv([lambda: env])
obs = env.reset()
print(obs)
save_dir = '../'
model = PPO2.load(save_dir + "PPO2_tutorial")


obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

