#!/usr/bin/env python3

from __future__ import division, print_function
import os
import sys
import numpy
import gym
import time
# from optparse import OptionParser
import argparse
import queue
import numpy as np
import csv  
import gym_minigrid
from policy import astar


def set_direction(env, start, end):
    # env.agent_dir 
    diff = (end[0] - start[0], end[1] - start[1])
    # for possibility in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        # if diff == possibility:
        #     # could have used if-else, but deduced a function for the same
        #     env.agent_dir = abs(possibility[0]*2 + possibility[1]*1 - 1)
        #     break
    if diff == (0, -1):
        # going up
        env.agent_dir = 3
    elif diff == (0, 1):
        # going down
        env.agent_dir = 1
    elif diff == (1, 0):
        # going right
        env.agent_dir = 0
    elif diff == (-1, 0):
        # going right
        env.agent_dir = 2
    else:
        raise ValueError("Difference between start and end is not valid")
    return env.agent_dir


def main():
    parser = argparse.ArgumentParser(description='Record trajectories with state, goal, action for train/test') 
     # the goal as blue or red and the grid for a-star agent')
    parser.add_argument('-s', '--split', default="test", choices=['train', 'test', 'infer'], help="train/test/infer")

    parser.add_argument("-e", "--env-name", dest="env_name",  
        help="gym environment to load",
        default='MiniGrid-TwoGoals-Random-16x16-v0'
    )
    parser.add_argument("-g", "--goal", choices=['blue', 'red'], default='blue')
    # parser.add_argument("-pos", "--goal-pos", type=list, dest='goal_pos', 
    #     help='specify goal position other than bottom corners, give (column, row) in the grid',
    #     default=None)
    parser.add_argument("-d", "--data-dir", dest="data_dir", default="trajectoryTwoGoals")
    # parser.add_argument("-gd", "--goal-dimension", dest="goal_dimension", type=int, 
        # help="Representation of the goal", default='100')

    # parser.add_argument("-gdist", "--goal-mutual-distance", dest="goal_mutual_distance", type=float, 
        # default = 0.1, help='Max 0.5, min 0.1 to remain reasonably withing 0 to 1?')

    parser.add_argument('-t', "--num-traj", dest="num_trajectories", type=int, default=10)

    parser.add_argument('--disable-render', action='store_false', help='Disable Rendering')
    
    args = parser.parse_args()
    print('args', args)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    FILENAME = os.path.join(args.data_dir, args.split)
        # _gd_{:0=4d}".format(args.goal_dimension))
        # + "_gdist_{:5f}".format(args.goal_mutual_distance))
    print(FILENAME)
    DATAFILENAME = FILENAME + ".npy"
    CSVFILENAME = FILENAME + ".csv"
    data = []

    if args.split == 'infer':
        num_trajectories = 1
    else:
        num_trajectories = args.num_trajectories
    if num_trajectories == 1:
        goals_list = ['red']
    else:
        goals_list = ['red', 'blue']


    # Load the gym environment
    # env_name_list = [
    # 'MiniGrid-TwoGoals-Random-16x16-v0',
    # # 'MiniGrid-TwoGoals-16x16-v0'
    # # 'MiniGrid-TwoGoals-Random-9x9-v0', 
    # # 'MiniGrid-TwoGoals-8x8-v0', 
    # # 'MiniGrid-TwoGoals-Random-6x6-v0',
    # # 'MiniGrid-TwoGoals-6x6-v0'
    # ]*num_trajectories
    env_name_list = [args.env_name] * num_trajectories


    path_array_list = []
    action_array_list = []
    goal_array_list = []


    for env_name in env_name_list:
        for goal_choice in goals_list:
            
            env = gym.make(env_name)

            def resetEnv():
                observation = env.reset()
                # print(observation)
                if hasattr(env, 'mission'):
                    print('Mission: %s' % env.mission)

            # maze = np.zeros((env.grid.height-2, env.grid.width-2))
            maze = np.zeros((env.grid.width,env.grid.height))

            resetEnv()
            def display():
                if args.disable_render:
                    renderer = None
                else:
                    renderer = env.render()
                return renderer
            display()
            # start = (env.agent_pos[0]-1, env.agent_pos[1]-1)
            start = (env.agent_pos[0], env.agent_pos[1])
            # print('start', start)
            # start = (0, 0)

            # HARD CODED goal positions 
            # ------------
            # TODO: change to get the goal position from env.
            # ------------
            if goal_choice == 'blue':
                # goal = (env.grid.height-3, env.grid.width-3)
                goal = (env.grid.width-2, env.grid.height-2)
                # goal_embedding = np.array([1 - args.goal_mutual_distance]*dim_goal_embedding)
                goal_indicator = 0
            elif goal_choice == 'red': 
                # goal = (0, env.grid.width-3)
                goal = (1, env.grid.height-2)
                # goal_embedding = np.array([1 + args.goal_mutual_distance]*dim_goal_embedding)
                goal_indicator = 1
            else:   
                raise ValueError

            # if goal_choice_pos is not None:
            #     goal = goal_choice_pos

            # print('goal', goal)
            path = astar.get_path(maze, start, goal)
            # print('path', path)

            # store the trajectory to view
            path_array = np.array(path,dtype=np.dtype('float')) / env.grid.height
            # print('path_array', path_array)
            goal_array = np.array(goal_indicator, dtype=np.dtype('float')) 
            # print('goal_array', goal_array) 
            action_array = np.zeros(len(path)-1)

            for i in range(len(path)-1):
                display()
                print('index', i, 'env.agent_pos',tuple(env.agent_pos))
                if env.front_pos[0] == goal[0] and env.front_pos[1] == goal[1]:
                    break
                step = set_direction(env, path[i], path[i+1])
                action_array[i] = step
                display()
                observation, reward, done, info = env.step(env.actions.forward)
                if not args.disable_render:
                    time.sleep(.5)  
                    # If the window was closed
                    if env.render().window == None:
                        break

            path_array_list.append(path_array)
            action_array_list.append(action_array)
            goal_array_list.append(goal_array)

            with open(CSVFILENAME, 'a') as f:
                writer = csv.writer(f)
                for i in range(len(path_array)-1):
                    writer.writerow((path_array[i][0], path_array[i][1], goal_array, action_array[i]))
                    data.append((path_array[i][0], path_array[i][1], goal_array, action_array[i]))
                


    with open(DATAFILENAME, 'wb') as f_data:
        np.save(f_data, np.array(data))
    # saving  for training neural network
    # with open('R_actions.npy', 'wb') as f_action:
    #     np.save(f_action, np.array(action_array_list))

    # with open('R_trajectories.npy', 'wb') as f_traj:
    #     np.save(f_traj, np.array(path_array_list))

    # with open('R_goals.npy', 'wb') as f_goal:
    #     np.save(f_goal, np.array(goal_array_list))



if __name__ == "__main__":
    main()
