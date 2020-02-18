from gym import spaces
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(self,
        top,
        size,
        doorPos
    ):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(
            topX + 1, topX + sizeX - 1,
            topY + 1, topY + sizeY - 1
        )

    def set_goals_pos(self, env, num_goals, set_goals=True):
        goalsPos = set()
        while set_goals:
            goalsPos.add(self.rand_pos(env))
            if len(goalsPos) == 2: #  int(num_goals):
                break
        for goal in goalsPos:
            env.grid.set(*goal, Box('red', toggletimes=3))

        return goalsPos


class HallwayWithVictims(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        width=25,
        height=25,
        random_door_pos=False,
    ):
        self.random_door_pos = random_door_pos
        super().__init__(width=width, height=height, max_steps=10*width*height)
        self.total_reward = 0
        self.num_goals = 0


    def _gen_grid(self, width, height, sideway_length=4):
        # Create the grid
        self.total_reward = 0
        self.grid = Grid(width, height)

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0+sideway_length, height-sideway_length):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        # Generate the surrounding walls and hall sideways
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height-1, Wall())
            if i >= sideway_length and i <= width-1-sideway_length:
                if i >= lWallIdx and i <= rWallIdx:
                    # self.grid.set(i, sideway_length, Lava())
                    # self.grid.set(i, height-1-sideway_length, Floor())
                    continue
                else:
                    self.grid.set(i, sideway_length, Wall())
                    self.grid.set(i, height-1-sideway_length, Wall())

        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width-1, j, Wall())
            if j >= sideway_length and j <= height-1-sideway_length:
                self.grid.set(sideway_length, j, Wall())
                self.grid.set(width-1-sideway_length, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0+sideway_length, height-sideway_length):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # Room splitting walls
        num_rooms = 3
        for n in range(0, num_rooms):
            j = n * (height - 2*sideway_length) // 3
            for i in range(0+sideway_length, lWallIdx):
                self.grid.set(i, j+ sideway_length, Wall())
            for i in range(rWallIdx, width-sideway_length):
                self.grid.set(i, j+ sideway_length, Wall())

            # Left corridor roooms
            # leftroomWallIndex = (sideway_length + lWallIdx - 2) // 2
            # for i in range(0+sideway_length, height-sideway_length):
            #     self.grid.set(leftroomWallIndex, i, Wall())

            # Right corridor rooms

            # Add Rooms to grid list
            roomW = 0 - sideway_length + lWallIdx + 1
            roomH = (height - 2*sideway_length) // 3 + 1

            # Do not sample door locations
            if not self.random_door_pos:
                if True:
                    #leftroomdoorpos =  (sideway_length, j + sideway_length + self._rand_int(1, roomH-1))
                    leftroomdoorpos = (lWallIdx,  j + sideway_length  + self._rand_int(1, roomH-1))
                    rightroomdoorpos = (rWallIdx, j + sideway_length + self._rand_int(1, roomH-1))
                else:
                    leftroomdoorpos = (lWallIdx,  j + sideway_length  + self._rand_int(1, roomH-1))
                    rightroomdoorpos = (width-1-sideway_length, j + sideway_length + self._rand_int(1, roomH-1))
            else:
                # Sample door locations from left or right
                if np.random.randint(2):
                    leftroomdoorpos = (lWallIdx,  j + sideway_length  + self._rand_int(1, roomH-1))
                else:
                    leftroomdoorpos = (lWallIdx,  j + sideway_length  + self._rand_int(1, roomH-1))

                if np.random.randint(2):
                    rightroomdoorpos = (rWallIdx, j + sideway_length + self._rand_int(1, roomH-1))
                else:
                    rightroomdoorpos = (width-1-sideway_length, j + sideway_length + self._rand_int(1, roomH-1))

            self.rooms.append(Room(
                (0+sideway_length, j+sideway_length),
                (roomW, roomH),
                [leftroomdoorpos]
            ))
            self.rooms.append(Room(
                (rWallIdx, j + sideway_length),
                (roomW, roomH),
                [rightroomdoorpos]
            ))

        # Choose goal or victim positions in the rooms
        # Let each room have at most three goals
        max_goals_per_room = 3
        self.num_goals = 0
        for i, room in enumerate(self.rooms):
            num_goals = self._rand_int(0, max_goals_per_room)
            # num_goals = (i) % 3
            set_goals_RV = bool(np.random.randint(2))
            goalsPosList = room.set_goals_pos(self, num_goals, set_goals=set_goals_RV)
            self.num_goals += (2 * int(set_goals_RV))

        #print('goals', self.num_goals)

        # # Choose one random room to be locked
        # lockedRoom = self._rand_elem(self.rooms)
        # lockedRoom.locked = True
        # goalPos = lockedRoom.rand_pos(self)
        # self.grid.set(*goalPos, Goal())

        # # Assign the door colors
        colors = set(COLOR_NAMES)
        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            room.color = color
            for doorPos in room.doorPos:
                if room.locked:
                    self.grid.set(*doorPos, Door(room.color, is_locked=True))
                else:
                    self.grid.set(*doorPos, Door(room.color))

        # # Select a random room to contain the key
        # while True:
        #     keyRoom = self._rand_elem(self.rooms)
        #     if keyRoom != lockedRoom:
        #         break
        # keyPos = keyRoom.rand_pos(self)
        # self.grid.set(*keyPos, Key(lockedRoom.color))

        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(width//2, height-2),
            size=(2, 2)
        )

        # Generate the mission string
        self.mission = (
            'Triage as many victims as possible ...'
        )


    def step(self, action):
        fwd_cell = self.grid.get(*self.front_pos)
        obs, reward, done, info = MiniGridEnv.step(self, action)
        fwd_cell_after = self.grid.get(*self.front_pos)
        # import ipdb; ipdb.set_trace()

        if action == self.actions.toggle and fwd_cell is not None and fwd_cell.type == 'box':
            # If toggled, then the box should be none or not a box
            if fwd_cell_after is None:
                reward = 1
            elif fwd_cell_after.type != 'box':
                reward = 1
            # Add total reward
            self.total_reward += reward
            if self.total_reward >= self.num_goals:
                done = True

        return obs, reward, done, info

class HallwayWithVictimsRandom(HallwayWithVictims):
    def __init__(self, width=25, height=25):
        super().__init__(width=width, height=height, random_door_pos=True)


register(
    id='MiniGrid-HallwayWithVictims-v0',
    entry_point='gym_minigrid.envs:HallwayWithVictims'
)

register(
    id='MiniGrid-HallwayWithVictims-v1',
    entry_point='gym_minigrid.envs:HallwayWithVictimsRandom'
)
