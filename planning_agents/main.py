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
from aiagents import *

parser = argparse.ArgumentParser()
parser.add_argument('--agenttype', type=int, required=True, help='Preemptive=0, Selective=1, SelectiveV1=2, MixedTime=3')
parser.add_argument('--fullobs', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=10)
parser.add_argument('--procrastination_frac', type=float, default=0.8)
parser.add_argument('--astardelta', type=int, default=0)
parser.add_argument('--save', type=int, default=1)
args = parser.parse_args()

OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'white' : 6,
}

def minimap_to_rgb(nmap):
    '''
    Given minimap, convert to rgb
    '''
    IDX_2_RGB = {
        0: [217, 217, 217],
        1: [0, 0, 0],
        2: [165, 42, 42],
        4: [0, 100, 0],
        2: [173, 216, 230],
        7 : [50, 50, 170],
        8: [-1, -1, -1],
        9: [255, 15, 0],
        5: [255, 201, 102],
        10: [0, 255, 255],
    }
    COL_2_RGB = {
       0: [255, 0, 0],
       1: [0, 255, 0],
       6: [255, 255, 255],
       4: [255, 255, 0],
    }
    def idx2rgb(idx, channel=0):
        return IDX_2_RGB[idx][channel]

    def col2rgb(idx, channel=0):
        return COL_2_RGB[idx][channel]

    obj, col, state = nmap.T
    H, W = obj.shape
    out = np.zeros((H, W, 3))
    for i in range(3):
        out[..., i] = np.vectorize(idx2rgb)(obj, i)
    y, x = np.where(out[..., 0] < 0)
    for i in range(3):
        out[y, x, i] = np.vectorize(col2rgb)(col[y, x], i)
    return out / 255.0


# Set value of save here
save=args.save
print1 = print
if save:
    print = lambda *x: None
#########################
## Main code
#########################
env = gym.make("MiniGrid-NumpyMapMinecraftUSAR-v0")
env.agent_view_size = 17
env = wrappers.AgentExtraInfoWrapper(env)

#agent = PreEmptiveAgentLeft(env) if args.agenttype == 0 else PreEmptiveAgentRight(env)
#agent = SelectiveAgentLeft(env) if args.agenttype == 0 else SelectiveAgentRight(env)
#agent = SelectiveAgentV1Left(env) if args.agenttype == 0 else SelectiveAgentV1Right(env)
#agent = MixedTimeAgentLeft(env, procrastination_frac=args.procrastination_frac) if args.agenttype == 0 else MixedTimeAgentRight(env, procrastination_frac=args.procrastination_frac)
agtype = args.agenttype
if agtype == 0:
    agent = PreEmptiveAgentLeft(env)
elif agtype == 1:
    agent = SelectiveAgentLeft(env)
elif agtype == 2:
    agent = SelectiveAgentV1Left(env)
elif agtype == 3:
    agent = MixedTimeAgentLeft(env, procrastination_frac=args.procrastination_frac)
elif agtype == 4:
    agent = MixedProximityAgentLeft(env, astardelta=args.astardelta)
print(agent)

# Init env and action
obs = env.reset()
info = {}
agent.reset(env.get_map(), obs)
act = agent.predict(obs, info, 0)

#agent.update(obs)
#print(obs['pos'], obs['dir'])
print(OBJECT_TO_IDX)

## Save trajectories here
expert_data = []
current_episode_actions = dict(obs=[], act=[], rew=[], fullmap=None)
current_episode_actions['fullmap'] = env.get_full_map()
episodes = 0
num_steps = 0
rew = 0

total_rewards = []

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
        print1("Episode {} done in {} steps with reward {}".format(episodes + 1, num_steps, np.sum(current_episode_actions['rew'])))
        total_rewards.append(np.sum(current_episode_actions['rew']))
        episodes += 1
        num_steps = 0
        obs = env.reset()
        agent.reset(env.get_map(), obs)
        # Add this episode data to all episodes
        expert_data.append(current_episode_actions)
        current_episode_actions = dict(obs=[], act=[], rew=[], fullmap=None)
        current_episode_actions['fullmap'] = env.get_full_map()

    act = agent.predict(obs, info, rew)
    #print(obs['pos'], obs['dir'])

    if not save:
        plt.clf()
        plt.subplot(121)
        #img = env.render('rgb_array')
        img = env.get_full_map()
        img = minimap_to_rgb(img)
        plt.imshow(img)
        plt.title('Ground truth')

        plt.subplot(122)
        #plt.imshow(agent.get_belief_map_image().transpose(1, 0, 2))
        #plt.imshow(agent.get_prob_map(['door']).T, 'jet')
        #plt.imshow(agent.get_lastvisited_map().T, 'gray')
        #plt.imshow(agent.get_frontiers_map().T, 'gray')
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
    tr = total_rewards
    print1(tr)
    print1("Reward stats: Mean: {}, Std: {}, Min: {}, Max: {}".format(np.mean(tr), np.std(tr), np.min(tr), np.max(tr)))
    print1(agent)
    filename = input('Enter filename: ')
    filename += '.pkl'

    with open(filename, 'wb') as fi:
        pkl.dump(expert_data, fi)
        print1("Saved to {}".format(filename))

