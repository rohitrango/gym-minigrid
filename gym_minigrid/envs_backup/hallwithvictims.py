from gym import spaces
from matplotlib import pyplot as plt
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

TOGGLETIMES = {
        'red': 5,
        'yellow': 3,
        'green': 3,
}
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
        self.victimcolor = None
        self.locked = False

        self.redcount = 0
        self.greencount = 0

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
            rand_color = 'red' if np.random.rand() < 0.5 else 'yellow'
            if rand_color == 'red':
                self.redcount += 1
            else:
                self.greencount += 1
            # If new red victim comes in or previous red victim present, keep it red
            if rand_color == 'red' or self.victimcolor == 'red':
                self.victimcolor = 'red'
            else:
                self.victimcolor = 'yellow'
            env.grid.set(*goal, Goal(rand_color, toggletimes=TOGGLETIMES[rand_color], triage_color='green'))

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
        dog=True,
    ):
        self.random_door_pos = random_door_pos
        self.dog = dog
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
            set_goals_RV = bool(np.random.randint(2)) if i != 3 else True
            goalsPosList = room.set_goals_pos(self, num_goals, set_goals=set_goals_RV)
            self.num_goals += (2 * int(set_goals_RV))

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
        fwd_color = fwd_cell.color if fwd_cell is not None else None
        obs, reward, done, info = MiniGridEnv.step(self, action)
        fwd_cell_after = self.grid.get(*self.front_pos)
        # import ipdb; ipdb.set_trace()

        # Get bark info here
        if fwd_cell_after is not None and fwd_cell_after.type == 'door' \
                and not fwd_cell_after.is_open and self.dog:
            for room in self.rooms:
                if self.front_pos[0] == room.doorPos[0][0] \
                and self.front_pos[1] == room.doorPos[0][1]:
                    # Set up bark value
                    info['dog'] = room.victimcolor if room.victimcolor is not None else 0
                    if info['dog'] in ['red', 'yellow']:
                        info['red'] = room.redcount
                        info['yellow'] = room.greencount
                    break

        # Toggle a goal
        if action == self.actions.toggle and fwd_cell is not None and fwd_cell.type == 'goal':
            # If toggled, then the goal should be none or not a goal
            if fwd_cell_after.type == 'goal' and fwd_cell_after.color == 'green' and fwd_color != 'green':
                reward = 1
            elif fwd_cell_after.type != 'goal':
                reward = 1
            # Add total reward
            self.total_reward += reward
            if self.total_reward >= self.num_goals:
                done = True

        return obs, reward, done, info

class HallwayWithVictimsRandom(HallwayWithVictims):
    def __init__(self, width=25, height=25):
        super().__init__(width=width, height=height, random_door_pos=True)

###############################
## SAR map from numpy array
###############################
index_mapping = {
    9: 'empty',
    8: 'agent',
    1: 'agent',
    2: 'door',
    4: 'dummywall',
    5: 'lava',
    6: 'key',
    7: 'box',
    3: 'box',
    0: 'wall',
    10: 'obstacle',
    -1: 'obstacle'
}
color_mapping = {
    9: '',
    8: '',
    1: '',
    2: 'green',
    4: 'grey',
    5: '',
    6: 'yellow',
    7: '',
    3: '',
    0: '',
    10: 'green',
    -1: 'red'
}

class HallwayWithVictimsSARmap(HallwayWithVictims):
    def __init__(
        self,
        width=50,
        height=50,
        random_door_pos=False,
        dog=True,):
        # init the grid
        self.num_goals = 0
        super().__init__(width=width, height=height, random_door_pos=random_door_pos,dog=dog)

    def _gen_grid(self, *args, **kwargs):
        # Here goes our map
        #mapfile = '../gym_minigrid/grid.npy'
        mapfile = '/home/rohitrango/gym-minigrid/gym_minigrid/grid.npy'
        data = np.load(mapfile).T
        w, h = data.shape

        self.grid = Grid(w, h)
        self.grid.wall_rect(0, 0, w, h)
        for i in range(1, w-1):
            for j in range(1, h-1):
                idx = int(data[i, j])
                entity_name = index_mapping[idx]
                color = color_mapping[idx]

                if entity_name == '':
                    continue
                elif entity_name in ['wall']:
                    self.put_obj(Wall(), i, j)
                elif entity_name in ['obstacle']:
                    #self.put_obj(Box('green', toggletimes=1, triage_color=None), i, j)
                    pass
                elif entity_name == 'box':
                    color = 'red' if np.random.rand() < 0.5 else 'yellow'
                    self.put_obj(Goal(color, toggletimes=TOGGLETIMES[color], triage_color='green'), i, j)
                    self.num_goals += 1
                elif entity_name == 'door':
                    self.put_obj(Door(color='blue'), i, j)
                elif entity_name == 'lava':
                    self.put_obj(Lava(), i, j)
                elif entity_name == 'agent':
                    # Place the agent here
                    self.agent_pos = (i, j)

        self.place_agent()
        self.mission = 'Triage all victims'

    def get_dogvalue(self, door_pos, agent_pos):
        # Given agent pos and door pos, estimate the room, get all victims in it
        # and return a dog bark
        # If the room is too big, then return nothing
        grid = self.grid
        room = np.zeros((grid.width, grid.height))
        dirx = door_pos[0] - agent_pos[0]
        diry = door_pos[1] - agent_pos[1]
        firstpos = np.array((door_pos[0] + dirx, door_pos[1] + diry))
        # Get nodes
        nodes = [firstpos]
        while len(nodes) != 0:
            room[nodes[0][0], nodes[0][1]] = 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nbr = nodes[0] + (dx, dy)
                    cell = grid.get(*nbr)
                    if cell is not None and cell.type in ['door', 'wall']:
                        continue
                    if room[nbr[0], nbr[1]] == 0:
                        nodes.append(nbr)
            nodes = nodes[1:]

        # This is too big of a room
        #plt.figure()
        #plt.imshow(room.T)
        #plt.show()
        max_room_size = 200
        if room.sum() > max_room_size:
            print("Too big room (you shouldn't see many of these warnings...)")
            return None, None, None

        # Not too big
        # Check for red and green victims inside these rooms
        x, y = np.where(room == 1)
        redcount = greencount = 0
        for xc, yc in zip(x, y):
            cell = grid.get(xc, yc)
            if cell is not None and cell.type == 'box':
                if cell.color == 'red':
                    redcount += 1
                elif cell.color == 'green':
                    greencount += 1
        # Give out a bark now
        if redcount > 0:
            return 'red', redcount, greencount
        elif greencount > 0:
            return 'green', redcount, greencount
        return 0, 0, 0



    def step(self, action):
        fwd_cell = self.grid.get(*self.front_pos)
        obs, reward, done, info = MiniGridEnv.step(self, action)
        fwd_cell_after = self.grid.get(*self.front_pos)
        # import ipdb; ipdb.set_trace()

        # Get bark info here
        if fwd_cell_after is not None and fwd_cell_after.type == 'door' \
                and not fwd_cell_after.is_open and self.dog:
            # Get a dog feedback here
            info['dog'], r, g = self.get_dogvalue(self.front_pos, self.agent_pos)
            info['red'] = r
            info['green'] = g

        if action == self.actions.toggle and fwd_cell is not None and fwd_cell.type == 'box':
            # If toggled, then the box should be none or not a box
            if fwd_cell_after is None:
                reward = 0
            elif fwd_cell_after.type == 'box' and fwd_cell_after.color == 'yellow':
                reward = 0
            elif fwd_cell_after.type != 'box':
                reward = 0
            # Add total reward
            self.total_reward += reward
            if self.total_reward >= self.num_goals:
                done = True

        return obs, reward, done, info

register(
    id='MiniGrid-HallwayWithVictims-v0',
    entry_point='gym_minigrid.envs:HallwayWithVictims'
)

register(
    id='MiniGrid-HallwayWithVictims-SARmap-v0',
    entry_point='gym_minigrid.envs:HallwayWithVictimsSARmap'
)

register(
    id='MiniGrid-HallwayWithVictims-v1',
    entry_point='gym_minigrid.envs:HallwayWithVictimsRandom'
)
