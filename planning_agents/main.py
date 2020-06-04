'''
Base agents that observe partial environments and take actions based on preferences
'''
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import gym
from heapq import *
import gym_minigrid
from scipy.ndimage import zoom
from gym_minigrid import wrappers
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
import argparse
from aiagents import PreEmptiveAgent

parser = argparse.ArgumentParser()
parser.add_argument('--agenttype', type=int, required=True, help='Preemptive=0, Scouring>=1')
parser.add_argument('--fullobs', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--save', type=int, default=1)
args = parser.parse_args()

# Set value of save here
save=args.save
print1 = print
if save:
    print = lambda *x: None
#########################
## Main code
#########################
env = gym.make("MiniGrid-NumpyMapMinecraftUSAR-v0")
env.agent_view_size = 9
env = wrappers.AgentExtraInfoWrapper(env)

agent = PreEmptiveAgent(env) if args.agenttype == 0 else None
print(agent)

# Init env and action
obs = env.reset()
info = {}
agent.reset(env.get_map(), obs)
act = agent.predict(obs, info)

#agent.update(obs)
#print(obs['pos'], obs['dir'])
print(OBJECT_TO_IDX)

## Save trajectories here
expert_data = []
current_episode_actions = dict(obs=[], act=[], rew=[])
episodes = 0
num_steps = 0

while episodes < args.num_episodes:
    #act = int(input("Enter action "))
    #agent.update(obs)
    #act = agent.predict(obs)
    if args.fullobs:
        fullmap = env.get_full_map()
    else:
        fullmap = obs
    current_episode_actions['obs'].append(fullmap)
    current_episode_actions['act'].append(act)
    obs, rew, done, info = env.step(act)
    current_episode_actions['rew'].append(rew)
    if info != {}:
        print(info)
    num_steps += 1
    if done:
        print1("Episode {} done in {} steps".format(episodes + 1, num_steps))
        episodes += 1
        num_steps = 0
        obs = env.reset()
        agent.reset(env.get_map(), obs)
        # Add this episode data to all episodes
        if len(current_episode_actions['act']) <= 1000:
            expert_data.append(current_episode_actions)
        current_episode_actions = dict(obs=[], act=[], rew=[])

    act = agent.predict(obs, info)
    #print(obs['pos'], obs['dir'])

    if not save:
        plt.clf()
        plt.subplot(121)
        img = env.render('rgb_array')
        plt.imshow(img)
        plt.title('Ground truth')

        plt.subplot(122)
        #plt.imshow(agent.get_belief_map_image().transpose(1, 0, 2))
        #plt.imshow(agent.get_prob_map(['door']).T, 'jet')
        plt.imshow(agent.get_entropy().T, 'jet')
        plt.title('Agent\'s belief')
        plt.suptitle(agent.get_dogml_info())

        '''
        plt.subplot(132)
        plt.imshow(agent.get_prob_map(['box', 'goal']).T, 'jet')
        plt.title('Victim')

        plt.subplot(133)
        plt.imshow(agent.get_prob_map(['wall']).T, 'jet')
        plt.title('Map')

        plt.subplot(224)
        plt.imshow(agent.get_entropy().T, 'jet')
        plt.title('Certainty')
        '''
        plt.draw()
        plt.pause(.05)

if save:
    # Get filename
    print1(agent)
    filename = input('Enter filename: ')
    filename += '.pkl'

    with open(filename, 'wb') as fi:
        pkl.dump(expert_data, fi)
        print1("Saved to {}".format(filename))

