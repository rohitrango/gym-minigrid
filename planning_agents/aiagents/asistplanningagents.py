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
        self.victimcolor = ['yellow', 'green']
        self.agent_pos = None
        self.agent_dir = None
        self._subgoal = None
        self._current_plan = None

        # Other variables (for Player to make tradeoffs)
        self.time = 0
        self.victimlifetime = None
        self.numyellow = None
        self.numgreen = None
        self.reward_green = None
        self.reward_yellow = None

        # Tradeoff variables
        self.boxcost = 0.5

        # Dog maximum likelihood graph
        self.dogmlgraph = {'green': [0, 0, 0], 'yellow': [0, 0, 0]}

    def is_victim_in_priority(self, colnum):
        # Given a color number, check if the victim is in priority
        # check from victimcolor
        for col in self.victimcolor:
            if colnum == COLOR_TO_IDX[col]:
                return True
        return False


    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I < 0)
        I[idxneg] = I[idxneg] * 0.5
        return np.abs(I)


    def check_safe_plan(self):
        # TODO: Check if the next step is safe, if not then delete the existing plan
        dirvec = NUM_TO_DIR[self.agent_dir]
        fwd_cell = np.array(self.agent_pos) + dirvec
        # Get object, color, state in front of it
        fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)

        # If there is a current plan
        if len(self.plan) > 0:
            # If plan is explore, then check if the belief is 0, we have explored that point then
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

            # If the plan is to move forward, check for potential obstacles (because we may not have seen those things but now we do)
            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = self.agent_pos + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava', 'key']:
                    self.plan = []
                elif fwd_obj == 'door':
                    if fwd_state == STATE_TO_IDX['closed']:
                        self.plan.append('toggle')
                elif fwd_obj == 'goal':
                    if self.is_victim_in_priority(fwd_color):
                        self.plan = []
                        self.plan.append('toggle')
                elif fwd_obj == 'box':
                    self.plan.append('toggle')

        # If there is no plan currently, probably at a door or victim that we want to triage
        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = np.array(self.agent_pos) + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'goal':
                if self.is_victim_in_priority(fwd_color):
                    self.plan.append('toggle')
            elif fwd_obj == 'box':
                self.plan.append('toggle')

        # If we're exploring, we might want to ditch the plan if we found some victims (this is after all safe checks)
        if self._current_plan == 'explore':
            # If you're exploring and see something
            A = self.agent_view_size
            maxprob = 0
            for color in self.victimcolor:
                maxprob = maxprob + self.get_prob_map(['goal'], zoom_factor=1, color=color)
            maxprob = np.max(maxprob)
            if maxprob >= 0.7:
                #print("Preemptively ditching plan for goal")
                self.plan = []


    def reset(self, nmap, obs):
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

        self.time = 0

        # Update other info
        self._init_subagent(obs)


    def get_max_belief(self):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A] + 0
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
        belief = self.belief[A:-A, A:-A] + 0
        img = np.zeros((belief.shape[0], belief.shape[1], 3))
        # The walls are white in color
        img += self.get_prob_map(['wall'])[:, :, None]
        # Color code victims
        img[:, :, 1] += self.get_prob_map(['goal'], color='green')
        img[:, :, 0] += self.get_prob_map(['goal'], color='yellow')
        img[:, :, 1] += self.get_prob_map(['goal'], color='yellow')
        img[:, :, 0] += 0.5*self.get_prob_map(['goal'], color='white')
        img[:, :, 1] += 0.5*self.get_prob_map(['goal'], color='white')
        img[:, :, 2] += 0.5*self.get_prob_map(['goal'], color='white')
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
        #dist = np.sqrt((hh - x)**2 + (ww - y)**2) / belief.shape[0]
        dist = (self.leakyabs(hh - x, 'x') + self.leakyabs(ww - y, 'y')) / belief.shape[0]
        dist = dist.T

        # Multiply with frontier
        ent = ent * self.get_frontiers_map()
        # get location of subgoal
        # x, y = self._subgoal
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
        rval = self.dogmlgraph['yellow']
        gval = self.dogmlgraph['green']
        rr = np.around(rval[1]/rval[0], 2) if rval[0] > 0 else 'unknown'
        rg = np.around(rval[2]/rval[0], 2) if rval[0] > 0 else 'unknown'

        gr = np.around(gval[1]/gval[0], 2) if gval[0] > 0 else 'unknown'
        gg = np.around(gval[2]/gval[0], 2) if gval[0] > 0 else 'unknown'

        stri = 'Two barks: P(green victim) = {}, P(yellow victim) = {} \n'.format(rg, rr)
        stri += 'One bark: P(green victim) = {}, P(yellow victim) = {}\n'.format(gg, gr)
        stri += 'Current plan: {}'.format(self._current_plan)
        return stri


    def update(self, obs, info):
        # Given the observation, update your belief
        self.time += 1
        img = obs['image']
        pos = obs['pos']
        agdir = obs['dir']
        bark = obs.get('bark', -1)

        # Save the values for planning ahead
        self.agent_pos = np.array(pos)
        self.agent_dir = agdir
        A = self.agent_view_size

        # TODO: Update likelihood info if some info is present
        if bark > 0 and False:
            y = int(bark == 2)
            g = int(bark == 1)
            self.dogmlgraph[key][0] += 1
            self.dogmlgraph[key][1] += y
            self.dogmlgraph[key][2] += g

        # Create a dummy victim ahead if you see some info
        # TODO
        if bark >= 0 and False:
            #dirvec = NUM_TO_DIR[self.agent_dir]
            #fwd_cell = self.agent_pos + dirvec
            #fwd_cell = fwd_cell + dirvec
            # Check for door nearby
            for dirvec in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                fwd_cell = self.agent_pos + dirvec
                fx, fy = fwd_cell
                if self.belief[A+fx, A+fy, OBJECT_TO_IDX['door']-1] > 0.5:
                    fwd_cell = fwd_cell + dirvec
                    break
            assert np.abs(fwd_cell - self.agent_pos).sum() == 2

            # Get room area
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

            # Given the bark number, update belief (only where entropy is high)
            y, x = np.where(room == 1)
            pr = self.belief[y, x]
            entr = (-pr * np.log(pr + 1e-100)).sum(-1)
            entr = (entr > 1e-2)
            y, x = y[entr], x[entr]

            barkcolor = 'yellow' if bark == 2 else 'green'
            if bark > 0:
                self.belief[y, x] = 0
                self.colors[y, x] = 0
                self.belief[y, x, OBJECT_TO_IDX['goal']-1] = 1
                self.colors[y, x, COLOR_TO_IDX[barkcolor]] = 1
            elif bark == 0:
                self.belief[y, x] = 0
                self.colors[y, x] = 0
                self.belief[y, x, OBJECT_TO_IDX['empty']-1] = 1


        # Now, update the belief from the observation that we saw
        topX, topY, botX, botY = self.get_bounds(pos, agdir)
        h, w = img.shape[:2]
        img[w//2, h-1, 0] = 0  # if we cant trust the object on agent's position in the observation
        for i in range(agdir + 1):
            img = self.rotate_right(img)

        # Forget some information as per the forgetting parameter
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

        # update visited as well
        A = self.agent_view_size
        self.lastvisited[topX + x + A, topY + y + A] = 1


    def get_lastvisited_map(self):
        A = self.agent_view_size
        visited = self.lastvisited[A:-A, A:-A]
        return visited


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


    def predict(self, obs, info, rew=None):
        self._update_from_reward(rew)
        self.update(obs, info)
        self.check_safe_plan()
        if self.plan == []:
            #print("Updating plan")
            self.update_plan()
        try:
            #if self.plan[-1] != 'toggle':
                #if np.random.rand() < 0.4:
                    #return ACTION_TO_NUM_MAPPING['toggle']
            return ACTION_TO_NUM_MAPPING[self.plan.pop()]
        except:
            print("Returning done")
            return ACTION_TO_NUM_MAPPING['done']


    def get_location_from_preference(self, prefstr):
        # prefstr is a string preference containing multiple color info, etc.
        A = self.agent_view_size
        pref = prefstr.split(' ')[0]
        extra = prefstr.split(' ')[1:] # contains extra parameters of the preference

        if pref in ['goal', 'box', 'door']:
            # Search for a high confidence goal according to belief
            goalcode = OBJECT_TO_IDX[pref] - 1
            probgoal = self.belief[A:-A, A:-A, goalcode] + 0
            # Dont get gray victims
            if pref == 'goal':
                # If nothing specified, go after yellow and green victims
                if extra == []:
                    colors = [COLOR_TO_IDX['yellow'], COLOR_TO_IDX['green']]
                else:
                    # Go after colors that are specified
                    colors = [COLOR_TO_IDX[x] for x in extra]
                # Multiply with the colors
                probgoal *= (self.colors[A:-A, A:-A, colors]).sum(-1)

            x, y = np.where(probgoal > 0.7)
            if len(x) == 0:
                return None
            # Check for unopened doors only [DEPRECATE THIS]
            if pref == 'door':
                doorstate = np.argmax(self.objstates[A+x, A+y], 1)
                doorstate = np.where(doorstate != STATE_TO_IDX['open'])
                x, y = x[doorstate], y[doorstate]
                if len(x) == 0:
                    return None
            # Get the one with min index
            print('Choosing from {} doors'.format(len(x)))
            #minidx = np.argmin(np.abs(x - self.agent_pos[0]) + np.abs(y - self.agent_pos[1]))
            # Take astar distance from each of them
            minidx = self._get_minpath_idx(x, y)
            return [x[minidx], y[minidx]]

        elif pref == 'explore':
            entropy = -self.belief * np.log(self.belief + 1e-100) + 0
            entropy = entropy[A:-A, A:-A].mean(2)
            entropy[self.agent_pos[0], self.agent_pos[1]] = 0
            # Get shape
            H, W = entropy.shape
            # Have a map based on spatial distance
            xx, yy = np.arange(H), np.arange(W)
            xx, yy = np.meshgrid(xx, yy)
            dist = self.leakyabs(xx - self.agent_pos[0], 'x') + self.leakyabs(yy - self.agent_pos[1], 'y')
            dist = dist.T * 1.0
            dist /= H
            entropydist = entropy / (1 + dist)
            frontier_map = self.get_frontiers_map()
            # Use full entropy map or masked one (if using full entropy map, then just take argmax)
            fullent = True
            if np.max(entropydist * frontier_map) > 0:
                entropydist = frontier_map * entropydist
                # TODO: make this part fast
                fullent = True
            # Sample from it
            N = entropy.reshape(-1).shape[0]
            try:
                #p = entropydist/entropydist.sum()
                #goal = np.random.choice(np.arange(N), p=p.reshape(-1))
                #goal = np.argmax(entropydist)
                #x, y = goal//H, goal%H
                if fullent:
                    goal = np.argmax(entropydist)
                    x, y = goal//H, goal%H
                else:
                    x, y = np.where(entropydist > 0)
                    idx = self._get_minpath_idx(x, y)
                    x, y = x[idx], y[idx]
                return [x, y]
            except:
                return None


    def _get_minpath_idx(self, x, y):
        '''
        Get the index which has minimum path length
        '''
        pathlength = []
        for _x, _y in zip(x, y):
            path = self.astar((_x, _y))
            path = list(filter(lambda x: x == 'forward', path))
            pathlength.append(len(path))
        return np.argmin(pathlength)


    def get_frontiers_map(self):
        # Get frontiers map
        A = self.agent_view_size
        visited = self.get_lastvisited_map()
        # Get neighbours to set frontiers
        nbr = visited * 0
        # Check for neighbours validity
        classes = ['wall']
        classes = [OBJECT_TO_IDX[x]-1 for x in classes]
        obstacle = 1 - self.belief[A:-A, A:-A, classes].sum(-1)
        # Get neighbours
        for i in [-1, 1]:
            nbr[1:] += (visited[:-1] * obstacle[:-1] + 0)
            nbr[:-1] += (visited[1:] * obstacle[1:] + 0)
            nbr[:, 1:] += (visited[:, :-1] * obstacle[:, :-1] + 0)
            nbr[:, :-1] += (visited[:, 1:] * obstacle[:, 1:] + 0)
        frontier = (nbr > 0) & (visited == 0)
        return frontier


    def update_plan(self):
        # Update a new plan
        assert self.plan == []
        # Check for preferences, and return a location to sample from
        loc = None
        preferences = self._get_preferences()
        for pref in preferences:
            loc = self.get_location_from_preference(pref)
            if loc is not None:
                self._current_plan = pref if not pref.startswith('goal') else 'Victim'
                break

        # The location should not be empty
        assert loc is not None
        self._subgoal = loc
        #print("Plan is {}".format(self._current_plan))
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

        # If both are same then return a default
        if src[0] == dst[0] and src[1] == dst[1]:
            action_seq = ['left', 'left', 'forward', 'left', 'left', 'forward']
            return action_seq[::-1]

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
                # obscost is a very high cost for crashing into lava, wall or key objects
                obscost = belief[nbr[0], nbr[1], [OBJECT_TO_IDX['lava']-1, OBJECT_TO_IDX['wall']-1, OBJECT_TO_IDX['key']-1]].sum()
                obscost = int(obscost > 0.7)*1e50
                # get a cost for crossing a box too because you have to toggle it
                boxcost = belief[nbr[0], nbr[1], [OBJECT_TO_IDX['box']-1]].sum() * self.boxcost
                gc = node[1] + 1 + obscost + boxcost
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
                assert last == src, (src, dst, last)
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

    def _init_subagent(self, obs):
        # Extra comamnds for subagent (use as per your agent = should be not implmented for the base class)
        raise NotImplementedError

    def _get_preferences(self):
        raise NotImplementedError

    def _update_from_reward(self, rew):
        if rew is None or self.reward_green is None or self.reward_yellow is None:
            return None

        if rew == self.reward_green:
            self.numgreen -= 1
        elif rew == self.reward_yellow:
            self.numyellow -= 1

