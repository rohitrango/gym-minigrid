'''
Contains agents that save the first victim they encounter without regard for anything else
'''
from .asistplanningagents import *

class SelectiveAgent(PlanAgent):
    '''
    This agent will triage yellow ones till time is up and then save green victims
    '''
    def _init_subagent(self, obs):
        self.time = obs['time']
        self.victimlifetime = obs['victimlifetime']
        self.numyellow = obs['numyellow']
        self.numgreen = obs['numgreen']
        self.reward_green = obs['reward_green']
        self.reward_yellow = obs['reward_yellow']

        # Only save yellow victims now
        self.victimcolor = ['yellow']


    def _get_preferences(self):
        # This is where we should focus on yellow ones
        if self.time <= self.victimlifetime and self.numyellow > 0:
            return ['goal yellow', 'explore']
        else:
            self.victimcolor = ['yellow', 'green']
            return ['goal green', 'explore']


class SelectiveAgentV1(PlanAgent):
    '''
    This agent will triage yellow ones till time is up and then save green victims as it sees fit
    '''
    def _init_subagent(self, obs):
        self.time = obs['time']
        self.victimlifetime = obs['victimlifetime']
        self.numyellow = obs['numyellow']
        self.numgreen = obs['numgreen']
        self.reward_green = obs['reward_green']
        self.reward_yellow = obs['reward_yellow']

        # Only save yellow victims now
        self.victimcolor = ['yellow']


    def _get_preferences(self):
        # This is where we should focus on yellow ones
        A = self.agent_view_size
        belief = self.belief[A:-A, A:-A] + 0
        ent = -belief * np.log(1e-100 + belief)
        ent = ent.sum(2)

        if self.time <= self.victimlifetime and self.numyellow > 0:
            return ['goal yellow', 'explore']
        else:
            self.victimcolor = []
            # get unexplored area
            unexplored_area = np.sum(ent > 1e-3)
            # Also get empty, box, goal area (which is explored area)
            explored_area = (belief[:, :, [OBJECT_TO_IDX[x]-1 for x in ['empty', 'box', 'goal']]].sum(-1) > 0.7).sum()
            # Get green victims who are explored
            prob = belief[:, :, OBJECT_TO_IDX['goal'] - 1] * self.colors[A:-A, A:-A, COLOR_TO_IDX['green']]
            explored_green_victims = np.sum(prob > 0.7)
            unexplored_green_victims = self.numgreen - explored_green_victims
            # Expected reward for exploration v/s exploitation
            explorereward = 1.0 * unexplored_green_victims #/ unexplored_area
            exploitreward = 1.0 * explored_green_victims #/ explored_area
            print('Seen: {}, Not seen: {}, total: {}'.format(exploitreward, explorereward, self.numgreen))
            if exploitreward > explorereward:
                self.victimcolor = ['green']
                return ['goal green', 'explore']
            return ['explore', 'goal green']


class SelectiveAgentLeft(SelectiveAgent):
    '''
    Bias to the top left
    '''
    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I < 0)
        I[idxneg] = I[idxneg] * 0.5
        return np.abs(I)


class SelectiveAgentRight(SelectiveAgent):
    '''
    Bias to the bottom right
    '''
    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I > 0)
        I[idxneg] = I[idxneg] * 0.5


class SelectiveAgentV1Left(SelectiveAgentV1):
    '''
    Bias to the top left
    '''
    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I < 0)
        I[idxneg] = I[idxneg] * 0.5
        return np.abs(I)


class SelectiveAgentV1Right(SelectiveAgentV1):
    '''
    Bias to the bottom right
    '''
    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I > 0)
        I[idxneg] = I[idxneg] * 0.5
        return np.abs(I)
        return np.abs(I)
