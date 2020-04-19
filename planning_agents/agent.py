'''
Base agents that observe partial environments and take actions based on preferences
'''
import numpy as np
from matplotlib import pyplot as plt
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
        self.preferences = ['goal', 'box', 'door', 'explore']
        self.agent_pos = None
        self.agent_dir = None
        self._subgoal = None
        self._current_plan = None


    def reset(self):
        self.belief = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numobjects)) / self.numobjects
        # Keep track of last visited
        self.lastvisited = np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        # Capture color and states
        self.colors = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numcolors)) / self.numcolors
        self.objstates = np.ones((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size, self.numstates)) / self.numstates

        self.plan = []
        self.agent_pos = None
        self.agent_dir = None
        self._subgoal = None
        self._current_plan = None

    def get_max_belief(self):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A]
        return np.argmax(belief, 2)

    def get_prob_map(self, classes, zoom_factor=4):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A] + 0
        classidx = list(map(lambda x: OBJECT_TO_IDX[x]-1, classes))
        prob = belief[:, :, classidx].sum(2)
        if zoom_factor > 1:
            prob = zoom(prob, zoom_factor, order=1)
            prob[0, 0] = 1
        return prob

    @property
    def subgoal(self):
        return self._subgoal

    def get_entropy(self):
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A]
        ent = -belief * np.log(1e-100 + belief)
        ent = ent.sum(2)
        ent = -zoom(ent, 4)
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


    def update(self, obs):
        # Given the observation, update your belief
        img = obs['image']
        pos = obs['pos']
        agdir = obs['dir']

        # Save the values for planning ahead
        self.agent_pos = pos
        self.agent_dir = agdir

        topX, topY, botX, botY = self.get_bounds(pos, agdir)
        h, w = img.shape[:2]
        img[w//2, h-1, 0] = 0
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
            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = self.agent_pos + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava']:
                    self.plan = []
                elif fwd_obj == 'door':
                    self.plan = []
                    if fwd_state == STATE_TO_IDX['closed']:
                        self.plan.append('toggle')
                elif fwd_obj == 'box':
                    self.plan = []
                    self.plan.insert(0, 'toggle')
                elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                    self.plan = []
                    self.plan.append('toggle')


        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = self.agent_pos + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'box':
                self.plan = []
                self.plan.insert(0, 'toggle')
            if fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
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


    def predict(self, obs):
        self.update(obs)
        self.check_safe_plan()
        if self.plan == []:
            self.update_plan()
        try:
            return ACTION_TO_NUM_MAPPING[self.plan.pop()]
        except:
            print("Returning done")
            return ACTION_TO_NUM_MAPPING['done']


    def get_location_from_preference(self, pref):
        A = self.agent_view_size
        victimidx = [COLOR_TO_IDX['red'], COLOR_TO_IDX['yellow']]
        if pref in ['goal', 'box', 'door']:
            # Search for a high confidence goal according to belief
            goalcode = OBJECT_TO_IDX[pref] - 1
            probgoal = self.belief[A:-A, A:-A, goalcode] + 0
            if pref == 'goal':
                colors = self.colors[A:-A, A:-A, victimidx].sum(2) + 0
                probgoal = probgoal * colors

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
            dist = np.sqrt((xx - self.agent_pos[0])**2 + (yy - self.agent_pos[1])**2)
            dist /= H
            entropydist = entropy / (1 + dist)
            # Sample from it
            N = entropy.reshape(-1).shape[0]
            try:
                p = entropydist/entropydist.sum()
                goal = np.random.choice(np.arange(N), p=p.reshape(-1))
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
                self._current_plan = pref
                break

        # The location should not be empty
        assert loc is not None
        self._subgoal = loc
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
                if visited[nbr[0], nbr[1]]:
                    continue
                visited[nbr[0], nbr[1]] = 1
                # Not visited, time to visit this
                prob = belief[nbr[0], nbr[1], [OBJECT_TO_IDX['lava']-1, OBJECT_TO_IDX['wall']-1]].sum()
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
        if len(self.plan) > 0:
            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = self.agent_pos + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava']:
                    self.plan = []
                elif fwd_obj == 'door':
                    self.plan = []
                    if fwd_state == STATE_TO_IDX['closed']:
                        self.plan.append('toggle')
                elif fwd_obj == 'box':
                    self.plan = []
                    self.plan.insert(0, 'toggle')
                elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                    self.plan = []
                    self.plan.append('toggle')


        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = self.agent_pos + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'box':
                self.plan = []
                self.plan.insert(0, 'toggle')
            elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                self.plan = []
                self.plan.append('toggle')


        if self._current_plan == 'explore':
            # If you're exploring and see something
            A = self.agent_view_size
            maxprob = self.get_prob_map(['box', 'goal'], zoom_factor=1)
            maxprob = np.max(maxprob)
            # Check for unopened doors
            doorprob = self.get_prob_map(['door'], zoom_factor=1)
            doorprob *= (self.objstates[A:-A, A:-A, STATE_TO_IDX['closed']])
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
        self.preferences = ['door', 'explore', 'goal', 'box']
        self.opendoors = 0

    def check_safe_plan(self):
        if len(self.plan) > 0:
            if self.plan[-1] == 'forward':
                dirvec = NUM_TO_DIR[self.agent_dir]
                fwd_cell = self.agent_pos + dirvec
                # Get object, color, state
                fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
                if fwd_obj in ['wall', 'lava']:
                    self.plan = []
                elif fwd_obj == 'door':
                    self.plan = []
                    if fwd_state == STATE_TO_IDX['closed']:
                        self.plan.append('toggle')
                        self.opendoors += 1
                        # Check if all doors are open, no need to explore anymore now
                        if self.opendoors >= 6:
                            self.preferences.remove('explore')
                            self.preferences.append('explore')

                elif fwd_obj == 'box':
                    self.plan = []
                    self.plan.insert(0, 'toggle')
                elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                    self.plan = []
                    self.plan.append('toggle')


        elif len(self.plan) == 0:
            # Toggle door as long as it takes
            dirvec = NUM_TO_DIR[self.agent_dir]
            fwd_cell = self.agent_pos + dirvec
            # Get object, color, state
            fwd_obj, fwd_color, fwd_state = self.query_cell(fwd_cell)
            if fwd_obj == 'box':
                self.plan = []
                self.plan.insert(0, 'toggle')
            elif fwd_obj == 'goal' and fwd_color in VICTIMCOLORS:
                self.plan = []
                self.plan.append('toggle')

        if self._current_plan == 'explore':
            # If you're exploring and see a door
            A = self.agent_view_size
            # Check for unopened doors
            doorprob = self.get_prob_map(['door'], zoom_factor=1)
            doorprob *= (self.objstates[A:-A, A:-A, STATE_TO_IDX['closed']])
            if np.max(doorprob) >= 0.7:
                #print("Preemptively ditching plan for door")
                self.plan = []


#########################
## Main code
#########################
env = gym.make('MiniGrid-FourRooms-v0')
env = gym.make('MiniGrid-HallwayWithVictims-v0')
#env = gym.make('MiniGrid-HallwayWithVictimsAndFire-v0')
#env = gym.make('MiniGrid-Empty-Random-10x10-v0')
env = wrappers.AgentExtraInfoWrapper(env)

#agent = PlanAgent(env)
agent = PreEmptiveAgent(env)
#agent = ScouringAgent(env)
obs = env.reset()
act = agent.predict(obs)
#agent.update(obs)
#print(obs['pos'], obs['dir'])
print(OBJECT_TO_IDX)

## Save trajectories here
all_episode_actions = []
current_episode_actions = []
episodes = 0
num_steps = 0

while episodes < 1000:
    #act = int(input("Enter action "))
    #agent.update(obs)
    #act = agent.predict(obs)
    obs, rew, done, info = env.step(act)
    num_steps += 1
    current_episode_actions.append(act)
    if done:
        print("Episode {} done in {} steps".format(episodes + 1, num_steps))
        episodes += 1
        num_steps = 0
        obs = env.reset()
        agent.reset()
        # Add this episode data to all episodes
        all_episode_actions.append(current_episode_actions)
        current_episode_actions = []

    act = agent.predict(obs)
    #print(obs['pos'], obs['dir'])

    if 1:
        plt.clf()
        plt.subplot(221)
        img = env.render('rgb_array')
        plt.imshow(img)

        plt.subplot(222)
        plt.imshow(agent.get_prob_map(['box', 'goal']).T, 'jet')
        plt.title('Victim')

        plt.subplot(223)
        plt.imshow(agent.get_prob_map(['wall']).T, 'jet')
        plt.title('Map')

        plt.subplot(224)
        plt.imshow(agent.get_entropy().T, 'jet')
        plt.title('Certainty')
        plt.draw()
        plt.pause(0.001)


# Get filename
print(agent)
filename = input('Enter filename: ')
filename += '.csv'

with open(filename, 'wt') as fi:
    lines = []
    for act in all_episode_actions:
        stract = ','.join(list(map(lambda x: str(x), act))) + "\n"
        lines.append(stract)
    fi.writelines(lines)
