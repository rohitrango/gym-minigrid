import math
import operator
from functools import reduce
from queue import deque

import numpy as np
import gym
from gym import error, spaces, utils
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from .minigrid import CELL_PIXELS

class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*tile_size, self.env.height*tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        return env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return full_grid

class FullyObsOneHotWrapper(gym.core.ObservationWrapper):
    """
    Convert the fully observed wrapper into a one hot tensor
    """
    def __init__(self, env, drop_color=False, keep_classes=None, flatten=True):
        #assert 'FullyObsWrapper' in env.__class__.__name__
        super().__init__(env)
        # Number of classes
        if not keep_classes:
            keep_classes = list(OBJECT_TO_IDX.keys())
        keep_classes.sort(key=lambda x: OBJECT_TO_IDX[x])
        # Save number of classes and find new mapping
        self.num_classes = len(keep_classes)
        # Keep a mapping from old to new mapping so that it becomes easier to map
        # to one hot
        self.object_to_new_idx = dict()
        for idx, k in enumerate(keep_classes):
            self.object_to_new_idx[OBJECT_TO_IDX[k]] = idx

        # Number of colors
        if drop_color:
            self.num_colors = 0
        else:
            self.num_colors = len(COLOR_TO_IDX)
        self.num_states = 4

        self.N = self.num_classes + self.num_colors + self.num_states

        # Define shape of the new environment
        self.obsshape = list(self.env.observation_space.shape[:2])
        if flatten:
            self.obsshape = np.prod(self.obsshape)
            shape = (self.obsshape * self.N, )
        else:
            raise NotImplementedError

        self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=shape,
                dtype='uint8',
            )

    def observation(self, obs):
        #obs = obs.reshape(-1)
        # Get one hot vector
        onehotclass = np.zeros((self.obsshape, self.num_classes), dtype=np.uint8)
        onehotcolor = np.zeros((self.obsshape, self.num_colors), dtype=np.uint8)
        onehotstate = np.zeros((self.obsshape, self.num_states), dtype=np.uint8)
        rangeobs = np.arange(self.obsshape)

        classes = obs[:, :, 0].reshape(-1)
        classes = np.vectorize(self.object_to_new_idx.__getitem__)(classes)
        onehotclass[rangeobs, classes] = 1

        # Go for color
        if self.num_colors > 0:
            colors = obs[:, :, 1].reshape(-1)
            onehotcolor[rangeobs, colors] = 1

        states = obs[:, :, 2].reshape(-1)
        onehotstate[rangeobs, states] = 1

        # Concat along the number of states dimension
        onehotobs = np.concatenate([onehotclass, onehotcolor, onehotstate], 1)
        return onehotobs.reshape(-1)


class AppendActionWrapper(gym.core.Wrapper):
    """
    Append the previous actions taken
    """
    def __init__(self, env, K):
        super().__init__(env)
        # K is the number of actions (including present)
        # size is the number of one hot vector
        self.K = K
        self.actsize = env.action_space.n
        self.history = deque([np.zeros(self.actsize) for _ in range(self.K)])
        self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.env.observation_space.shape[0] + self.actsize*self.K, ),
                dtype='uint8'
            )

    def reset(self, **kwargs):
        self.history = deque([np.zeros(self.actsize) for _ in range(self.K)])
        obs = self.env.reset(**kwargs)
        actall = np.concatenate(self.history)
        actall = actall.astype(np.uint8)
        # Append it to obs
        obs = np.concatenate([obs, actall])
        return obs


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # get one hot action
        act = np.zeros((self.actsize))
        act[action] = 1
        # update history
        self.history.popleft()
        self.history.append(act)
        actall = np.concatenate(self.history)
        actall = actall.astype(np.uint8)
        # Append it to obs
        obs = np.concatenate([obs, actall])
        return obs, reward, done, info


class GoalPolicyWrapper(gym.core.GoalEnv):
    """
    Encode a goal policy based on whether the agent reached the goal or not
    This is for simple navigation based goals only
    """
    def __init__(self, env, ):
        self.env = env
        assert isinstance(self.env, FullyObsOneHotWrapper)
        self.observation_space = gym.spaces.Dict({
                'observation': env.observation_space,
                'achieved_goal': env.observation_space,
                'desired_goal': env.observation_space,
            })
        self.action_space = env.action_space

    def _get_goals(self, Obs):
        # Create achieved and desired goals
        agentidx = self.env.object_to_new_idx[OBJECT_TO_IDX['agent']]
        emptyidx = self.env.object_to_new_idx[OBJECT_TO_IDX['empty']]
        goalidx = self.env.object_to_new_idx[OBJECT_TO_IDX['goal']]
        # Init the goals
        obs = Obs.reshape(self.env.obsshape, -1)
        achieved = obs + 0
        desired  = obs + 0
        # For achieved, just erase the goal
        achieved[:, goalidx] = 0
        # For desired, find the goal and replace by agent.
        # Replace the agent with empty
        agent_pos = np.where(desired[:, agentidx] > 0)
        goal_pos = np.where(desired[:, goalidx] > 0)

        desired[agent_pos, agentidx] = 0
        desired[agent_pos, emptyidx] = 1

        desired[goal_pos, goalidx] = 0
        desired[goal_pos, agentidx] = 1
        return achieved.reshape(-1), desired.reshape(-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        env = self.env
        while True:
            if hasattr(env, '_reward'):
                return env._reward()
            else:
                env = env.env

    def reset(self,):
        obs = self.env.reset()
        achieved, desired = self._get_goals(obs)

        return {
            'observation': obs,
            'achieved_goal': achieved,
            'desired_goal': desired,
        }

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        achieved, desired = self._get_goals(obs)
        obs_new =  {
            'observation': obs,
            'achieved_goal': achieved,
            'desired_goal': desired,
        }
        return obs_new, rew, done, info


class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class AgentViewWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent field of view size.
    """

    def __init__(self, env, agent_view_size=7):
        super(AgentViewWrapper, self).__init__(env)

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
