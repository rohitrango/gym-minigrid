
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class SimpleRoom(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        goal_num=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_num = goal_num

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        xc = width//2
        yc = height//2

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the goal depending on goal number
        # if goal is 0, place on left side, else on the
        # right side
        if self.goal_num == 0:
            xg, yg = np.random.randint(1, xc - 1), np.random.randint(1, yc - 1)
        elif self.goal_num == 1:
            xg, yg = np.random.randint(xc + 1, width). np.random.randint(1, yc - 1)
        self.grid.set(xg, yg, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

register(
    id='SimpleRoom-v0'
    entry_point='gym_minigrid.envs:SimpleRoom'
)
