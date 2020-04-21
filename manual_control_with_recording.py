#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
import pickle as pkl
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

# Store all observations here
all_observations = []
episode_data = dict(obs=[], act=[], rew=[])
G = 0

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size, hide_invisible=True)

    window.show_img(img)

def reset():
    global episode_data, all_observations, G
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    # store the previous episode here
    if len(episode_data['act']) > 0:
        all_observations.append(episode_data)
        episode_data = dict(obs=[], act=[], rew=[])
        G += 1

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption('Games completed: {}'.format(G))
    redraw(obs)

def step(action):
    global episode_data, all_observations
    fullmap = env.get_full_map()
    print(fullmap.shape)
    obs, reward, done, info = env.step(action)
    episode_data['obs'].append(fullmap)
    episode_data['act'].append(action)
    episode_data['rew'].append(reward)

    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    global episode_data, all_observations
    print('pressed', event.key)

    if event.key == 'escape':
        ## Save the trajectories here
        filename = input('Enter the filename: ')
        filename += '.pkl'
        with open(filename, 'wb') as fi:
            pkl.dump(all_observations, fi)

        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

# The actual main function
args = parser.parse_args()

env = gym.make(args.env)
if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
env = AgentExtraInfoWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
