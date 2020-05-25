import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        sizetop=None,
        extra=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.sizetop = sizetop
        self.extra = extra   # extra gives a choice mode (choose lava or goal)

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        if self.extra == 1:
            self.agent_start_pos = (1, np.random.randint(1, height-1))
        elif self.extra == 2:
            self.agent_start_dir = np.random.randint(4)

        # Generate the surrounding walls
        obj = Wall if self.extra < 2 else Lava
        self.grid.wall_rect(0, 0, width, height, obj_type=obj)

        # Place a goal square in the bottom-right corner
        if self.extra == 1:
            for i in range(1, height-1):
                self.put_obj(Lava(), width-2, i)
            self.put_obj(Goal(), width-2, np.random.randint(2, height-2))
        else:
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(size=self.sizetop)

        self.mission = "get to the green goal square"

class EmptyEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyEnv6x6Extra(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, extra=1)

class EmptyEnv6x6ExtraLava(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, extra=2)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyRandomEnv8x8(EmptyEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None)

class EmptyRandomEnv10x10(EmptyEnv):
    def __init__(self):
        super().__init__(size=10, agent_start_pos=None, sizetop=(4, 4))

class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v1',
    entry_point='gym_minigrid.envs:EmptyEnv6x6Extra'
)

register(
    id='MiniGrid-Empty-6x6-v2',
    entry_point='gym_minigrid.envs:EmptyEnv6x6ExtraLava'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv8x8'
)

register(
    id='MiniGrid-Empty-Random-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv10x10'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)
