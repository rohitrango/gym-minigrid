#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from policy import astar

# from gym_minigrid.wrappers import *
# from gym_minigrid.window import Window

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def objects_in_view(obs):
    view = {'door': [], 'box': []}
    l,w,c = obs.shape
    for i in range(l):
        for j in range(w):
            if obs[i, j, 2] == 4:
                view['door'].append((i, j))
            if obs[i, j, 2] == 7:
                view['box'].append((i, j))
                # view['box'] = {'pos':[(i, j)], 'color':COLOR_TO_IDX

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

def plan_astar(obs, curr_pos, target_pos):
    # Let us not use perception channels in obs
    maze = obs[:, :, ]

def plan_for_maxview(curr_pos):



args = parser.parse_args()

env = gym.make(args.env)
# env = gym.wrappers.Monitor(env, "recording")
# env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

# window = Window('gym_minigrid - ' + args.env)
# window.reg_key_handler(key_handler)

reset()

fwd_cell = self.grid.get(*self.front_pos)
view = objects_in_view(obs)

if view['box'][0] is not None:
    plan_astar(obs, curr_pos, view['box'][0])
    
elif view['door'][0] is not None:
    plan_astar(curr_pos, view['door'][0])

else:
    plan_for_maxview(curr_pos)

if fwd_cell.type == 'door' and fwd_cell.is_closed:
    env.step(self.action.toggle)

if fwd_cell.type == 'box':
    env.step(self.action.toggle)


# Blocking event loop
# window.show(block=True)
