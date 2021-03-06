'''
Base agents that observe partial environments and take actions based on preferences
'''
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import gym
from heapq import *
import gym_minigrid
from scipy.ndimage import zoom
from gym_minigrid import wrappers
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--agenttype', type=int, required=True, help='Preemptive=0, Scouring>=1')
parser.add_argument('--fullobs', type=int, default=0)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--save', type=int, default=1)
args = parser.parse_args()

#############
# Set value of save here
save=args.save
print1 = print
if save:
    print = lambda *x: None

DIR_TO_NUM = {
    (1, 0): 0,
    (0, -1): 3,
    (-1, 0): 2,
    (0, 1): 1,
}
NUM_TO_DIR = dict([(v, k) for k, v in DIR_TO_NUM.items()])

ACTION_TO_NUM_MAPPING = {
    'left': 0,
    'right': 1,
    'forward': 2,
    'toggle': 5,
    'done': 6,
}
IDX_TO_OBJECT = dict([(v, k) for k, v in OBJECT_TO_IDX.items()])
VICTIMCOLORS = ['red', 'yellow']
VICTIMCOLORS = list(map(lambda x: COLOR_TO_IDX[x], VICTIMCOLORS))

class PlanAgent:
    # Agent that has a plan! :D
    def __init__(self, env):
        # Given the env, get all the parameters like size and agent details
        # Env would be wrapped in a wrapper that gives agent location and direction
        self.env = env.env
        self.agent_view_size = self.env.agent_view_size
        self.width = self.env.width
        self.height = self.env.height

        self.epsilon = 0
        self.numobjects = len(OBJECT_TO_IDX) - 1
        self.numcolors = len(COLOR_TO_IDX)
        self.numstates = len(STATE_TO_IDX)

        # This is to capture belief
        self.belief = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numobjects)) / self.numobjects
        # Keep track of last visited
        self.lastvisited = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        # Capture color and states
        self.colors = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numcolors)) / self.numcolors
        self.objstates = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numstates)) / self.numstates

        self.plan = []
        #self.preferences = ['goal', 'door', 'explore']
        self.preferences = ['goal', 'explore']
        self.agent_pos = None
        self.agent_dir = None
        self._subgoal = None
        self._current_plan = None

        # Dog maximum likelihood graph
        self.dogmlgraph = {'red': [0, 0, 0], 'yellow': [0, 0, 0]}


    def reset(self, nmap):
        self.belief = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numobjects)) / self.numobjects
        # Keep track of last visited
        self.lastvisited = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        # Capture color and states
        self.colors = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numcolors)) / self.numcolors
        self.objstates = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numstates)) / self.numstates

        # Update wall locations
        for obj in ['wall', 'door']:
            y, x = np.where(nmap == OBJECT_TO_IDX[obj])
            y += self.agent_view_size
            x += self.agent_view_size
            self.belief[y, x] = 0
            self.belief[y, x, OBJECT_TO_IDX[obj]-1] = 1
            if obj == 'door':
                self.objstates[y, x, ] = 0
                self.objstates[y, x, STATE_TO_IDX['closed'] ] = 1

        self.plan = []
        self.agent_pos = None
        self.agent_dir = None
        self._subgoal = None
        self._current_plan = None


    def get_max_belief(self):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A]
        return np.argmax(belief, 2)

    def get_prob_map(self, classes, zoom_factor=1, color=None):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A] + 0
        colors = self.colors[A:-A, A:-A] + 0

        classidx = list(map(lambda x: OBJECT_TO_IDX[x]-1, classes))
        coloridx = 1
        if color is not None:
            coloridx = COLOR_TO_IDX[color]
            coloridx = colors[:, :, coloridx]
        prob = belief[:, :, classidx].sum(2) * coloridx
        if zoom_factor > 1:
            prob = zoom(prob, zoom_factor, order=1)
            prob[0, 0] = 1
        return prob

    def get_belief_map_image(self):
        # Get an image version of belief (because heatmaps are too confusing for non-statisticians)
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A]
        img = np.zeros((belief.shape[0], belief.shape[1], 3))
        # The walls are white in color
        img += self.get_prob_map(['wall'])[:, :, None]
        # Color code victims
        img[:, :, 1] += self.get_prob_map(['goal'], color='green')
        img[:, :, 0] += self.get_prob_map(['goal'], color='yellow')
        img[:, :, 1] += self.get_prob_map(['goal'], color='yellow')
        #img += self.get_prob_map(['goal'], color='yellow')[:, :, None] * 0.5
        # The doors are all blue
        doors = self.get_prob_map(['door'])
        img[:, :, 2] += doors

        img = np.minimum(1, img)
        return img

    @property
    def subgoal(self):
        return self._subgoal

    def get_entropy(self):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A]
        ent = -belief * np.log(1e-100 + belief)
        ent = ent.sum(2)
        #print(self._subgoal, self._current_plan)
        x, y = self.agent_pos
        h = np.arange(belief.shape[0])
        hh, ww = np.meshgrid(h, h)
        dist = np.sqrt((hh - x)**2 + (ww - y)**2) / belief.shape[0]
        dist = dist.T

        # get location of subgoal
        x, y = self._subgoal
        #ent[x, y] = np.max(ent)
        # zoom
        ent = zoom(ent/(1 + dist), 4, order=1)
        return ent

    def rotate_right(self, grid):
        newgrid = grid * 0
        # Swap elements
        H, W = grid.shape[:2]
        # Swap elements
        for i in range(H):
            for j in range(W):
                newgrid[i, j] = grid[j, H-1-i] + 0
        return newgrid

    def get_bounds(self, agent_pos, agent_dir):
        if agent_dir == 0:
            topX = agent_pos[0]
            topY = agent_pos[1] - self.agent_view_size // 2
        # Facing down
        elif agent_dir == 1:
            topX = agent_pos[0] - self.agent_view_size // 2
            topY = agent_pos[1]
        # Facing left
        elif agent_dir == 2:
            topX = agent_pos[0] - self.agent_view_size + 1
            topY = agent_pos[1] - self.agent_view_size // 2
        # Facing up
        elif agent_dir == 3:
            topX = agent_pos[0] - self.agent_view_size // 2
            topY = agent_pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size
        return (topX, topY, botX, botY)


    def get_dogml_info(self):
        rval = self.dogmlgraph['red']
        gval = self.dogmlgraph['yellow']
        rr = np.around(rval[1]/rval[0], 2) if rval[0] > 0 else 'unknown'
        rg = np.around(rval[2]/rval[0], 2) if rval[0] > 0 else 'unknown'

        gr = np.around(gval[1]/gval[0], 2) if gval[0] > 0 else 'unknown'
        gg = np.around(gval[2]/gval[0], 2) if gval[0] > 0 else 'unknown'

        stri = 'Two barks: P(red victim) = {}, P(yellow victim) = {} \n'.format(rr, rg)
        stri += 'One bark: P(red victim) = {}, P(yellow victim) = {}\n'.format(gr, gg)
        stri += 'Current plan: {}'.format(self._current_plan)
        return stri


    def update(self, obs, info):
        # Given the observation, update your belief
        img = obs['image']
        pos = obs['pos']
        agdir = obs['dir']

        # Save the values for planning ahead
        self.agent_pos = np.array(pos)
        self.agent_dir = agdir
        A = self.agent_view_size

        # Update victims if some info is present
        if info.get('dog') in ['red', 'yellow']:
            key = info['dog']
            r = int(info['red'] > 0)
            g = int(info['yellow'] > 0)
            self.dogmlgraph[key][0] += 1
            self.dogmlgraph[key][1] += r
            self.dogmlgraph[key][2] += g

        # Create a dummy victim ahead if you see some info
        if info.get('dog') in ['red', 'yellow'] or info.get('dog') == 0:
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = self.agent_pos + dirvec
            fwd_cell = fwd_cell + dirvec

            # Get room area
            print("Going inside....")
            room = np.zeros(self.belief.shape[:2])
            #print(room.shape)
            nodes = [fwd_cell+A]
            while len(nodes) != 0:
                room[nodes[0][0], nodes[0][1]] = 1
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if abs(dx) + abs(dy) != 1:
                            continue

                        nbr = nodes[0] + (dx, dy)
                        if self.belief[nbr[0], nbr[1], OBJECT_TO_IDX['wall']-1] > 0.5 \
                            or self.belief[nbr[0], nbr[1], OBJECT_TO_IDX['door']-1] > 0.5: \
                                continue
                        if room[nbr[0], nbr[1]] == 0:
                            nodes.append(nbr)
                nodes = nodes[1:]

            y, x = np.where(room == 1)
            '''
            self.belief[A + fwd_cell[0], A + fwd_cell[1]] = 0
            self.colors[A + fwd_cell[0], A + fwd_cell[1]] = 0
            self.belief[A + fwd_cell[0], A + fwd_cell[1], OBJECT_TO_IDX['goal']-1] = 1
            self.colors[A + fwd_cell[0], A + fwd_cell[1], COLOR_TO_IDX[info.get('dog')]] = 1
            '''
            if info.get('dog') in ['red', 'yellow']:
                self.belief[y, x] = 0
                self.colors[y, x] = 0
                self.belief[y, x, OBJECT_TO_IDX['goal']-1] = 1
                self.colors[y, x, COLOR_TO_IDX[info.get('dog')]] = 1
            elif info.get('dog') == 0:
                self.belief[y, x] = 0
                self.colors[y, x] = 0
                self.belief[y, x, OBJECT_TO_IDX['empty']-1] = 1

        topX, topY, botX, botY = self.get_bounds(pos, agdir)
        h, w = img.shape[:2]
        #img[w//2, h-1, 0] = 0
        #print(topX, topY, botX, botY)
        #print(self.agent_view_size)
        for i in range(agdir + 1):
            img = self.rotate_right(img)

        # Forget some information
        self.belief = (self.belief + self.epsilon)/(1 + self.numstates*self.epsilon)
        self.colors = (self.colors + self.epsilon)/(1 + self.numcolors*self.epsilon)
        self.objstates = (self.objstates + self.epsilon)/(1 + self.numobjects*self.epsilon)

        # Get seen portions only
        objs = img[:, :, 0]
        cols = img[:, :, 1]
        stas = img[:, :, 2]

        # Get locations
        x, y = np.where(objs > 0)
        vals = np.zeros((len(x), self.numobjects))
        vals[np.arange(len(x)), objs[x, y]-1] = 1
        # update colors and states
        colorvals = np.zeros((len(x), self.numcolors))
        colorvals[np.arange(len(x)), cols[x, y]] = 1

        statevals = np.zeros((len(x), self.numstates))
        statevals[np.arange(len(x)), stas[x, y]] = 1

        # update belief accordingly
        self.belief[topX + x + self.agent_view_size, topY + y + self.agent_view_size] = vals
        self.objstates[topX + x + self.agent_view_size, topY + y + self.agent_view_size] = statevals
        self.colors[topX + x + self.agent_view_size, topY + y + self.agent_view_size] = colorvals


    def check_safe_plan(self):

        # TODO: Check if the next step is safe, if not then delete the existing plan
        if len(self.plan) > 0:
            # If exploration is done, then move on
            if self._current_plan == 'explore':
                A = self.agent_view_size
                x, y = self._subgoal
                x += A
                y += A
                bel = self.belief[x, y]
                bel = np.sum(bel*(1 - bel))
                if bel < 1e-5:
                    self.plan = []
                    return

            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = self.agent_pos + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava']:
                    self.plan = []
                elif fwd_obj == 'door':
                    if fwd_state == STATE_TO_IDX['closed']:
                        #self.plan.insert(0, 'toggle')
                        self.plan.append('toggle')
                elif fwd_obj == 'box' :
                    #self.plan.insert(0, 'toggle')
                    self.plan.append('toggle')
                elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                    self.plan = []
                    self.plan.append('toggle')

        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = self.agent_pos + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'box' :
                #self.plan.insert(0, 'toggle')
                self.plan.append('toggle')
            elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                self.plan = []
                self.plan.append('toggle')


    def query_cell(self, pos):
        A = self.agent_view_size
        belief = self.belief[A + pos[0], A + pos[1]]
        color = self.colors[A + pos[0], A + pos[1]]
        state = self.objstates[A + pos[0], A + pos[1]]
        if max(belief) < 1.5/self.numobjects:
            return 'unknown', None, None
        idx = np.argmax(belief)
        cidx = np.argmax(color)
        sidx = np.argmax(state)
        return IDX_TO_OBJECT[idx + 1], cidx, sidx


    def predict(self, obs, info):
        self.update(obs, info)
        self.check_safe_plan()
        if self.plan == []:
            print("Updating plan")
            self.update_plan()
        try:
            #if self.plan[-1] != 'toggle':
                #if np.random.rand() < 0.4:
                    #return ACTION_TO_NUM_MAPPING['toggle']
            return ACTION_TO_NUM_MAPPING[self.plan.pop()]
        except:
            print("Returning done")
            return ACTION_TO_NUM_MAPPING['done']


    def get_location_from_preference(self, pref):
        A = self.agent_view_size
        if pref in ['goal', 'box', 'door']:
            # Search for a high confidence goal according to belief
            goalcode = OBJECT_TO_IDX[pref] - 1
            probgoal = self.belief[A:-A, A:-A, goalcode]
            # Dont get gray victims
            if pref == 'goal':
                probgoal *= (1 - self.colors[A:-A, A:-A, COLOR_TO_IDX['white']])

            x, y = np.where(probgoal > 0.7)
            if len(x) == 0:
                return None
            # Check for unopened doors only
            if pref == 'door':
                doorstate = np.argmax(self.objstates[A+x, A+y], 1)
                doorstate = np.where(doorstate != STATE_TO_IDX['open'])
                x, y = x[doorstate], y[doorstate]
                if len(x) == 0:
                    return None
            # Get the one with min index
            print('Choosing from {} doors'.format(len(x)))
            minidx = np.argmin(np.abs(x - self.agent_pos[0]) + np.abs(y - self.agent_pos[1]))
            return [x[minidx], y[minidx]]

        elif pref == 'explore':
            entropy = -self.belief * np.log(self.belief + 1e-100)
            entropy = entropy[A:-A, A:-A].mean(2)
            # Get shape
            H, W = entropy.shape
            # Have a map based on spatial distance
            xx, yy = np.arange(H), np.arange(W)
            xx, yy = np.meshgrid(xx, yy)
            dist = np.sqrt((xx - self.agent_pos[0])**2 + (yy - self.agent_pos[1])**2).T
            dist /= H
            entropydist = entropy / (1 + dist)
            # Sample from it
            N = entropy.reshape(-1).shape[0]
            try:
                p = entropydist/entropydist.sum()
                #goal = np.random.choice(np.arange(N), p=p.reshape(-1))
                goal = np.argmax(entropydist)
                x, y = goal//H, goal%H
                return [x, y]
            except:
                return None


    def update_plan(self):
        # Update a new plan
        assert self.plan == []
        # Check for preferences, and return a location to sample from
        loc = None
        for pref in self.preferences:
            loc = self.get_location_from_preference(pref)
            if loc is not None:
                self._current_plan = pref if pref != 'goal' else 'Victim'
                break

        # The location should not be empty
        assert loc is not None
        self._subgoal = loc
        print("Plan is {}".format(self._current_plan))
        # Make a plan to that location
        self.plan = self.astar(loc)
        return None


    def dist_func(self, a, b, mode='l1'):
        dist = 0
        if mode == 'l2':
            for t1, t2 in zip(a, b):
                dist += (t1 - t2)**2
            return np.sqrt(dist)
        elif mode == 'l1':
            for t1, t2 in zip(a, b):
                dist += np.abs(t1 - t2)
            return np.sqrt(dist)
        else:
            raise NotImplementedError


    def astar(self, dst):
        '''
        Run an A star algorithm to take a sequence of paths to take
        '''
        A = self.agent_view_size
        # Get source and dst
        src = list(self.agent_pos)
        agdir = self.agent_dir
        #print(src, dst)

        locations = []
        # Get belief and visited matrix
        belief = self.belief[A:-A, A:-A, :]
        colors = self.colors[A:-A, A:-A]
        visited = belief[:, :, 0] * 0
        visited[src[0], src[1]] = 1
        # Get parents
        parents = belief[:, :, :2] * 0 - 1
        # Make a heap for lowest to highest in terms of distance
        # heap elements will be of the form (f, g, pos)
        frontier = []
        heappush(frontier, (self.dist_func(src, dst), 0, src))
        # Expand the next node
        while len(frontier) != 0:
            node = heappop(frontier)
            loc = node[2]
            if loc == dst:
                break
            # Expand its neighbours
            #print('Expanding {}'.format(node))
            nbrs = self.get_neighbours(loc)
            for nbr in nbrs:
                try:
                    if visited[nbr[0], nbr[1]]:
                        continue
                except:
                    continue
                #plt.clf()
                #plt.subplot(211)
                #plt.imshow(visited)
                #plt.subplot(212)
                #plt.imshow(np.argmax(self.belief, 2))
                #plt.draw()
                visited[nbr[0], nbr[1]] = 1
                # Not visited, time to visit this
                prob = belief[nbr[0], nbr[1], [OBJECT_TO_IDX['lava']-1, OBJECT_TO_IDX['wall']-1, OBJECT_TO_IDX['key']-1]].sum()
                gc = node[1] + 1 + int(prob > 0.7)*1e50
                # Get f
                fc = gc + self.dist_func(nbr, dst)
                heappush(frontier, (fc, gc, nbr))
                # update parent of nbr
                parents[nbr[0], nbr[1]] = loc
        # We got the goal location, get the sequence of paths
        seq = [dst]
        while True:
            last = seq[-1]
            parent = list(parents[last[0], last[1]])
            parent = list(map(lambda x: int(x), parent))
            if parent == [-1, -1]:
                assert last == src
                break
            seq.append(parent)

        # Convert this sequence of locations into sequence of actions
        seq = seq[::-1]
        cur_dir = agdir
        action_seq = []
        for i in range(len(seq)-1):
            # Check for directions and take a step forward
            dst_dir = (seq[i+1][0] - seq[i][0], seq[i+1][1] - seq[i][1])
            dst_dir = DIR_TO_NUM[dst_dir]
            action_seq.extend(self.turn(cur_dir, dst_dir))
            action_seq.append('forward')
            # Update current direction
            cur_dir = dst_dir

        # Check for actions and map
        #print(seq)
        #print(action_seq, self.agent_dir)
        #self.finalmap = belief[:, :, 0]* 0
        #for i, s in enumerate(seq):
            #self.finalmap[s[0], s[1]] = i
        #plt.figure()
        #plt.imshow(self.finalmap.T)
        #plt.show()

        # Return sequences of actions
        return action_seq[::-1]

    def turn(self, cur_dir, dst_dir):
        steps = []
        while cur_dir != dst_dir:
            steps.append('right')
            cur_dir = (cur_dir+1)%4
        if len(steps) == 3:
            steps = ['left']
        return steps


    def get_neighbours(self, loc):
        nbrs = []
        x, y = loc[0], loc[1]
        for dx in [-1, 1]:
            if x+dx < 0 or x+dx >= self.height:
                continue
            nbrs.append([x+dx, y])
        for dy in [-1, 1]:
            if y+dy < 0 or y+dy >= self.width:
                continue
            nbrs.append([x, y+dy])
        return nbrs


class PreEmptiveAgent(PlanAgent):
    '''
    This agent stops whereever its going to when it sees a door or goal
    Two parts:
        If going to a door or goal, keep going
        If exploring and see a door/goal, then go to door/goal
    '''
    def check_safe_plan(self):
        # TODO: Check if the next step is safe, if not then delete the existing plan
        dirvec = NUM_TO_DIR[self.agent_dir]
        fwd_cell = np.array(self.agent_pos) + dirvec
        # Get object, color, state
        fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
        if fwd_obj != 'empty':
            print(fwd_obj, fwd_color)

        if len(self.plan) > 0:
            if self._current_plan == 'explore':
                A = self.agent_view_size
                x, y = self._subgoal
                x += A
                y += A
                bel = self.belief[x, y]
                bel = np.sum(bel*(1 - bel))
                if bel < 1e-5:
                    print("Wrong belief?")
                    self.plan = []
                    return

            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = self.agent_pos + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava']:
                    self.plan = []
                elif fwd_obj == 'door':
                    if fwd_state == STATE_TO_IDX['closed']:
                        self.plan.append('toggle')
                elif fwd_obj == 'goal':
                    if fwd_color != COLOR_TO_IDX['white']:
                        self.plan = []
                        self.plan.append('toggle')
                elif fwd_obj == 'box':
                    #self.plan.insert(0, 'toggle')
                    self.plan.append('toggle')

        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = np.array(self.agent_pos) + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'goal':
                if fwd_color != COLOR_TO_IDX['white']:
                    self.plan = []
                    self.plan.append('toggle')
            elif fwd_obj == 'box':
                #self.plan.insert(0, 'toggle')
                self.plan.append('toggle')

        if self._current_plan == 'explore':
            # If you're exploring and see something
            A = self.agent_view_size
            maxprob = self.get_prob_map(['goal'], zoom_factor=1, color='yellow') + self.get_prob_map(['goal'], zoom_factor=1, color='green')
            maxprob = np.max(maxprob)
            # Check for unopened doors
            #doorprob = self.get_prob_map(['door'], zoom_factor=1)
            #doorprob *= (self.objstates[A:-A, A:-A, STATE_TO_IDX['closed']])
            doorprob = [0]
            if maxprob >= 0.7:
                #print("Preemptively ditching plan for goal")
                self.plan = []
            elif np.max(doorprob) >= 0.7:
                #print("Preemptively ditching plan for door")
                self.plan = []


class ScouringAgent(PlanAgent):
    '''
    This agent minimizes overall entropy first and then goes to plan
    Two parts:
        If going to a door or goal, keep going
        If exploring and see a door/goal, then go to door/goal
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preferences = ['explore', 'goal','door']
        self.opendoors = 0

    def check_safe_plan(self):
        if len(self.plan) > 0:
            if self._current_plan == 'explore':
                A = self.agent_view_size
                x, y = self._subgoal
                x += A
                y += A
                bel = self.belief[x, y]
                bel = np.sum(bel*(1 - bel))
                if bel < 1e-5:
                    self.plan = []
                    return

            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = np.array(self.agent_pos) + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava']:
                    self.plan = []
                elif fwd_obj == 'door':
                    if fwd_state == STATE_TO_IDX['closed']:
                        self.plan.append('toggle')
                        self.opendoors += 1
                        # Check if all doors are open, no need to explore anymore now
                        if self.opendoors >= 6:
                            self.preferences.remove('explore')
                            self.preferences.append('explore')

                elif fwd_obj == 'goal':
                    if fwd_color != COLOR_TO_IDX['white']:
                        self.plan = []
                        self.plan.append('toggle')

                elif fwd_obj == 'box':
                    #self.plan.insert(0, 'toggle')
                    self.plan.append('toggle')


        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = np.array(self.agent_pos) + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'box':
                #self.plan.insert(0, 'toggle')
                self.plan.append('toggle')
            elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                self.plan = []
                self.plan.append('toggle')

        if self._current_plan == 'explore':
            # If you're exploring and see a door
            A = self.agent_view_size
            # Check for unopened doors
            doorprob = self.get_prob_map(['door'], zoom_factor=1)
            doorprob *= (self.objstates[A:-A, A:-A, STATE_TO_IDX['closed']])
            # TODO: Reduce probability to less than 1 to have this behavior
            if np.max(doorprob) >= 1.7:
                #print("Preemptively ditching plan for door")
                self.plan = []


#########################
## Main code
#########################
#env = gym.make('MiniGrid-HallwayWithVictims-v0')
#env = gym.make('MiniGrid-HallwayWithVictims-SARmap-v0')
env = gym.make("MiniGrid-NumpyMapMinecraftUSARRandomVictims-v0")
#env = gym.make("MiniGrid-NumpyMapMinecraftUSAR-v4")
env.agent_view_size = 9
env.dog = False
env = wrappers.AgentExtraInfoWrapper(env)

#agent = PlanAgent(env)
agent = PreEmptiveAgent(env) if args.agenttype == 0 else ScouringAgent(env)
#agent = ScouringAgent(env)
print(agent)

# Init env and action
obs = env.reset()
info = {}
agent.reset(env.get_map())
act = agent.predict(obs, info)

#agent.update(obs)
#print(obs['pos'], obs['dir'])
print(OBJECT_TO_IDX)

## Save trajectories here
expert_data = []
current_episode_actions = dict(obs=[], act=[], rew=[])
episodes = 0
num_steps = 0

while episodes < args.num_episodes:
    #act = int(input("Enter action "))
    #agent.update(obs)
    #act = agent.predict(obs)
    if args.fullobs:
        fullmap = env.get_full_map()
    else:
        fullmap = obs
    current_episode_actions['obs'].append(fullmap)
    current_episode_actions['act'].append(act)
    obs, rew, done, info = env.step(act)
    current_episode_actions['rew'].append(rew)
    if info != {}:
        print(info)
    num_steps += 1
    if done:
        print1("Episode {} done in {} steps".format(episodes + 1, num_steps))
        episodes += 1
        num_steps = 0
        obs = env.reset()
        agent.reset(env.get_map())
        # Add this episode data to all episodes
        if len(current_episode_actions['act']) <= 1000:
            expert_data.append(current_episode_actions)
        current_episode_actions = dict(obs=[], act=[], rew=[])

    act = agent.predict(obs, info)
    #print(obs['pos'], obs['dir'])

    if not save:
        plt.clf()
        plt.subplot(121)
        img = env.render('rgb_array')
        plt.imshow(img)
        plt.title('Ground truth')

        plt.subplot(122)
        #plt.imshow(agent.get_belief_map_image().transpose(1, 0, 2))
        #plt.imshow(agent.get_prob_map(['door']).T, 'jet')
        plt.imshow(agent.get_entropy().T, 'jet')
        plt.title('Agent\'s belief')
        plt.suptitle(agent.get_dogml_info())

        '''
        plt.subplot(132)
        plt.imshow(agent.get_prob_map(['box', 'goal']).T, 'jet')
        plt.title('Victim')

        plt.subplot(133)
        plt.imshow(agent.get_prob_map(['wall']).T, 'jet')
        plt.title('Map')

        plt.subplot(224)
        plt.imshow(agent.get_entropy().T, 'jet')
        plt.title('Certainty')
        '''
        plt.draw()
        plt.pause(.05)

if save:
    # Get filename
    print1(agent)
    filename = input('Enter filename: ')
    filename += '.pkl'

    with open(filename, 'wb') as fi:
        pkl.dump(expert_data, fi)
        print1("Saved to {}".format(filename))

