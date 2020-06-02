from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from pathlib import Path
import numpy as np

from gym_minigrid import preprocessing


RESOURCES_DIR = (Path(__file__).parent / './resources').resolve()

class NumpyMapMinecraftUSARv4(MiniGridEnv):

    def __init__(self, numpyFile=Path(RESOURCES_DIR, 'mapv4.npy'), agent_start_pos=[24, 25], agent_start_dir=2):
        self.numpyFile = numpyFile
        self.roomFile = Path(RESOURCES_DIR, 'tmp_grid_roominfo.npy')
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.index_mapping = {
                1: 'empty',
                2: 'door',
                #5: 'box',
                9: 'lava',
                10: 'agent',
                4: 'wall',
                30: 'wall',
                80: 'goal',
                81: 'goal',
                82: 'goal',
                83: 'goal',
                #255: 'box',
                5: 'empty',
                255: 'empty',
                -1: 'unseen',
            }
        self.color_mapping = {
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
            10: 'yellow',
            255: 'red',
            80: 'red',
            81: 'yellow',
            82: 'green',
            83: 'white',
            30: '',
        }
        self.toggletimes_mapping = {
            9: 0,
            8: 0,
            1: 0,
            2: 1,
            4: 0,
            5: 0,
            6: 1,
            7: 5,
            3: 5,
            0: 0,
            10: 3,
            30: 0,
            80: 0,
            81: 1,
            82: 1,
            83: 0,
            255: 2,
        }
        array = np.load(self.numpyFile)
        h, w = array.shape
        super().__init__(width=w, height=h, max_steps=1000, agent_view_size=7)


    def _get_filtered_map(self, grid):
        nmap = 0
        assert grid in self.index_mapping.values(), print(grid, self.index_mapping.values())
        for i, x in self.index_mapping.items():
            if x == grid:
                nmap = nmap + (self.array == i).astype(int)
        return nmap

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Create the grid
        self.array = np.load(self.numpyFile)
        self.array = preprocessing.preprocess_connected_components(self.array, self.index_mapping)
        # mask with wall where no access
        self.array[:, :12] = 4
        self.array[:36, :24] = 4

        # Save a wallgrid and doorgrid
        self.wallmap = self._get_filtered_map('wall') + self._get_filtered_map('unseen')
        self.doormap = self._get_filtered_map('door')

        # Put entities in the grid
        for mc_i in range(0, self.array.shape[0]):
            for mc_j in range(0, self.array.shape[1]):
                mg_i , mg_j = mc_i , mc_j
                entity_index = int(self.array[mc_i][mc_j])

                entity_name = self.index_mapping[entity_index]
                entity_color = self.color_mapping[entity_index]
                entity_toggletime = self.toggletimes_mapping[entity_index]

                if entity_name in ['empty']:
                    continue
                elif entity_name == 'agent':
                    self.agent_state_pos = (mc_j, mc_i)
                elif entity_name == 'unseen':
                    self.put_obj(Wall(), mg_j, mg_i)
                else:
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            # print(entity_index, entity_name, entity_color, entity_toggletime)
                            if entity_color != '':
                                self.put_obj(entity_class(color=entity_color), mg_j, mg_i)
                            else:
                                if entity_class == Goal:
                                    self.put_obj(entity_class(color=np.random.choice(['yellow', 'green'])), mg_j, mg_i)
                                else:
                                    self.put_obj(entity_class(), mg_j, mg_i)

        self.grid.wall_rect(0, 0, width, height)
        # Set agent position and directions
        self.agent_pos = self.agent_start_pos
        self.grid.set(*self.agent_start_pos, None)
        self.agent_dir = self.agent_start_dir
        # self.place_obj(Goal())
        self.mission = 'Triage the victims'

        # Load room information for easy querying later on
        #self.roomViews = self._get_door_to_room_mapping(np.load(self.roomFile))
        self.roomViews  = dict()


    def _get_door_to_room_mapping(self, roomboxes):
        dooridx = []
        for i, x in self.index_mapping.items():
            if x == 'door':
                dooridx.append(i)
        gridmap = self.array
        door_to_indices = dict()
        for box in roomboxes:
            x1, y1, x2, y2 = box
            # Given coordinates, find doors
            subarray = self.array[y1:y2+1, x1:x2+1]
            t = 0
            for i in dooridx:
                t = t + (subarray == i)
            Y, X = np.where(t > 0)
            Y += y1
            X += x1
            # Given all doors, we find the point outside the box
            for y, x in zip(Y, X):
                if y == y1:
                    door_to_indices[(x, y-1)] = self._get_roomcoordinates(box)
                elif y == y2:
                    door_to_indices[(x, y+1)] = self._get_roomcoordinates(box)
                elif x == x1:
                    door_to_indices[(x-1, y)] = self._get_roomcoordinates(box)
                elif x == x2:
                    door_to_indices[(x+1, y)] = self._get_roomcoordinates(box)
                else:
                    print(x, y, box)
                    raise NotImplementedError
        return door_to_indices


    def _get_roomcoordinates(self, box):
        # Given the box coordinates, find connected components
        x1, y1, x2, y2 = box
        visited = self.array * 0
        visited[y1+1:y2, x1+1:x2] = 1
        queue = []
        # Save points x, y
        y, x = np.where(visited)
        queue = list(zip(x, y))
        # Given queue elements, try to expand until u face walls or doors
        while len(queue) > 0:
            x, y = queue.pop(0)
            # Given this point, check for neighbouring points
            for nx, ny in [(x, y-1), (x, y+1), (x+1, y), (x-1, y)]:
                if visited[ny, nx] or self.wallmap[ny, nx] or self.doormap[ny, nx]:
                    continue
                # this is a non-visited door or key, add it to queue
                visited[ny, nx] = 1
                queue.append((nx, ny))
        # we're done with roomcoordinates, maybe show this
        y, x = np.where(visited)
        return (x, y)


    def colorbasedreward(self, color):
        if color == 'yellow':
            return 10
        elif color == 'green':
            return 5
        elif color == 'white':
            return 0
        else:
            NotImplementedError

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        cur_cell = tuple(self.agent_pos)

        bark = 0
        box = self.roomViews.get(cur_cell)
        if box is not None and action == self.actions.forward:
            x, y = box
            ele = self.array[y, x]
            idx = np.where((ele == 7).astype(int) + ele == 3)
            y1 = y[idx]
            x1 = x[idx]
            for x, y in zip(x1, y1):
                cell = self.grid.get(x, y)
                if cell is None or cell.type != 'goal':
                    print("SOMETHING IS WRONG, found this at {}, {}".format(x, y), cell)
                    raise NotImplementedError
                else:
                    color = cell.color
                    if color == 'yellow':
                        bark = 2
                    elif color == 'green' and bark < 2:
                        bark = 1

        obs['bark'] = bark

        if action == self.actions.forward or True:
            cur_cell = self.grid.get(*self.agent_pos)
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell != None and fwd_cell.type == 'goal':
                reward = self.colorbasedreward(fwd_cell.color)
                self.put_obj(Goal('white'), self.front_pos[0], self.front_pos[1])
                self.array[self.front_pos[1], self.front_pos[0]] = 9
                done = False
            elif cur_cell != None and cur_cell.type == 'goal':
                done = False
        return obs, reward, done, info



class NumpyMapMinecraftUSARRandomVictimsv4(NumpyMapMinecraftUSARv4):
    def __init__(self, num_victims_red=10, num_victims_green=20):
        self.num_victims_red = num_victims_red
        self.num_victims_green = num_victims_green
        super().__init__()

    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Create the grid
        self.array = np.load(self.numpyFile)
        self.array = preprocessing.preprocess_connected_components(self.array, self.index_mapping)

        for mc_i in range(0, self.array.shape[0]):
            for mc_j in range(0, self.array.shape[1] ):
                mg_i , mg_j = mc_i , mc_j
                entity_index = int(self.array[mc_i][mc_j])

                entity_name = self.index_mapping[entity_index]
                entity_color = self.color_mapping[entity_index]
                entity_toggletime = self.toggletimes_mapping[entity_index]

                if entity_name in ['agent', 'empty']:
                    continue
                elif entity_name == 'unseen':
                    self.put_obj(Wall(), mg_j, mg_i)
                else:
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            # print(entity_index, entity_name, entity_color, entity_toggletime)
                            if entity_name == 'goal':
                                # Skip goal i.e. victim placement
                                continue
                            elif entity_color != '':
                                self.put_obj(entity_class(color=entity_color), mg_j, mg_i)
                            else:
                                self.put_obj(entity_class(), mg_j, mg_i)

        for _ in range(self.num_victims_red):
            self.place_obj(Goal('yellow'))
        for _ in range(self.num_victims_green):
            self.place_obj(Goal('green'))

        self.agent_pos = self.agent_start_pos
        self.grid.set(*self.agent_start_pos, None)
        self.agent_dir = self.agent_start_dir
        # self.place_obj(Goal())
        self.mission = 'Triage the yellow and green victims.'


register(
    id='MiniGrid-NumpyMapMinecraftUSAR-v4',
    entry_point='gym_minigrid.envs:NumpyMapMinecraftUSARv4'
)

register(
    id='MiniGrid-NumpyMapMinecraftUSARRandomVictims-v4',
    entry_point='gym_minigrid.envs:NumpyMapMinecraftUSARRandomVictimsv4'
)
