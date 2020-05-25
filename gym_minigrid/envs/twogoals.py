from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class TwoGoalsEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        color1='yellow',
        color2='green'
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.color1 = color1
        self.color2 = color2
        self.step_count = 0
        self.max_goals = 2
        super().__init__(
            grid_size=size,
            max_steps=size*size,
            # Set this to True for maximum speed
            see_through_walls=True, 
            seed=np.random.randint(100000)
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corners
        self.put_obj(Goal(self.color1), width - 2, height - 2)
        self.put_obj(Goal(self.color2), 1, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green or red goal square"


    def reset(self):
        # Step count since episode start
        self.step_count = 0
        self.goal_count = 0

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Return first observation
        obs = self.gen_obs()
        return obs


    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        left_pos = self.left_pos
        right_pos = self.right_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Also get contents of cells to left and right
        left_cell = self.grid.get(*self.left_pos)
        right_cell = self.grid.get(*self.right_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True


        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell.type == 'goal' :
                fwd_cell.toggle(self, fwd_pos)
                if fwd_cell.color == 'green':
                    reward = 0.25 #. - self.
                    
                elif fwd_cell.color == 'yellow':
                    reward = 0.5

                self.goal_count += 1

        # Done action (not used by default)
        elif action == self.actions.done:
            done = True

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        if self.goal_count >= self.max_goals:
            reward += 1. - 0.9 * self.step_count/self.max_steps

            done=True

        obs = self.gen_obs()
        print('goal_count', self.goal_count)
        return obs, reward, done, {}


class TwoGoalsEnv5x5(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=5)

class TwoGoalsRandomEnv5x5(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class TwoGoalsEnv6x6(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=6)

class TwoGoalsRandomEnv6x6(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class TwoGoalsRandomEnv9x9(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=9, agent_start_pos=None)

class TwoGoalsEnv16x16(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=16)

class TwoGoalsRandomEnv16x16(TwoGoalsEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None)

register(
    id='MiniGrid-TwoGoals-5x5-v0',
    entry_point='gym_minigrid.envs:TwoGoalsEnv5x5'
)

register(
    id='MiniGrid-TwoGoals-Random-5x5-v0',
    entry_point='gym_minigrid.envs:TwoGoalsRandomEnv5x5'
)

register(
    id='MiniGrid-TwoGoals-6x6-v0',
    entry_point='gym_minigrid.envs:TwoGoalsEnv6x6'
)

register(
    id='MiniGrid-TwoGoals-Random-6x6-v0',
    entry_point='gym_minigrid.envs:TwoGoalsRandomEnv6x6'
)

register(
    id='MiniGrid-TwoGoals-8x8-v0',
    entry_point='gym_minigrid.envs:TwoGoalsEnv'
)

register(
    id='MiniGrid-TwoGoals-Random-9x9-v0',
    entry_point='gym_minigrid.envs:TwoGoalsRandomEnv9x9'
)

register(
    id='MiniGrid-TwoGoals-16x16-v0',
    entry_point='gym_minigrid.envs:TwoGoalsEnv16x16'
)

register(
    id='MiniGrid-TwoGoals-Random-16x16-v0',
    entry_point='gym_minigrid.envs:TwoGoalsRandomEnv16x16'
)

register(
    id='MiniGrid-TwoGoals-9x9-v0',
    entry_point='gym_minigrid.envs:TwoGoalsEnv9x9'
)
