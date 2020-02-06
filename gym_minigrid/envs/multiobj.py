from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MultiObj(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
        self,
        size=12,
        numObjs=4,
        num_rooms=3,
    ):
        # Number of objects
        self.numObjs = numObjs
        self.num_rooms = num_rooms
        # Keep track of number of items picked in order
        self.tracker_color = 0

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def add_door(self, i, j, door_idx=None, color=None, locked=None):
        """
        Add a door to a room, connecting it to a neighbor
        """

        color = self._rand_color()
        door = Door(color, is_locked=False)
        pos = (i, j)
        self.grid.set(*pos, door)

        return door, pos

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        num_rooms = self.num_rooms

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        # Save the door positions
        self.door_positions = []

        # Function for rejecting ball positions
        def is_near_door(grid, pos):
            # get all door positions
            doors = grid.door_positions
            for door in doors:
                dist = abs(door[0] - pos[0]) + abs(door[1] - pos[1])
                if dist <= 1:
                    return True
            return False

        # Create rooms
        for i in range(1, num_rooms):
            j = np.random.randint(1, height - 1)
            self.grid.vert_wall(i*height//num_rooms, 0)
            self.add_door(i*height//num_rooms, j)
            # Add it to list
            self.door_positions.append((i*height//num_rooms, j))

        types = ['ball']
        objs = []

        # For each object to be generated
        i = 0
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = IDX_TO_COLOR[i]
            i += 1

            if objType == 'ball':
                obj = Ball(objColor)
            else:
                raise NotImplementedError('wtf')

            self.place_obj(obj, reject_fn=is_near_door)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Generate the mission string
        self.mission = 'Pick up all the colors in an order'
        assert hasattr(self, 'mission')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        reward = 0

        if self.carrying:
            # Check if picked up in order
            if COLOR_TO_IDX[self.carrying.color] == self.tracker_color:
                self.tracker_color += 1
            if self.tracker_color == self.numObjs:
                reward = 1
                done = True
            # Delete that object
            self.carrying = None

        return obs, reward, done, info

class MultiObj5x5N2(MultiObj):
    def __init__(self):
        super().__init__(size=5, numObjs=2)

class MultiObj6x6N2(MultiObj):
    def __init__(self):
        super().__init__(size=6, numObjs=2)

register(
    id='MiniGrid-MultiObj-5x5-N2-v0',
    entry_point='gym_minigrid.envs:MultiObj5x5N2'
)

register(
    id='MiniGrid-MultiObj-6x6-N2-v0',
    entry_point='gym_minigrid.envs:MultiObj6x6N2'
)

register(
    id='MiniGrid-MultiObj-8x8-N3-v0',
    entry_point='gym_minigrid.envs:MultiObj'
)
