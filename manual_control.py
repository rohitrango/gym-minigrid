#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import numpy as np
import gym
import time
from optparse import OptionParser
from matplotlib import pyplot as plt
import gym_minigrid

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_option(
            '--size',
            type=int,
            default=20,
            )
    parser.add_option(
            '--goal_num',
            help='set goal number',
            type=int,
            default=0
            )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name, size=options.size,  goal_num=options.goal_num, )
    env.agent_view_size = options.size
    env.see_through_walls = True

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs = env.gen_obs()
        image, dir_ = obs['image'], obs['direction']
        image = image[:, :, 0]
        image = np.rot90(image, -1)[:, ::-1]
        print(image, dir_)
        #plt.imshow(image)
        #plt.show()
        obs, reward, done, info = env.step(action)
        print(obs['direction'])
        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
