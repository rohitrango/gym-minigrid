'''
Contains agents that save the first victim they encounter without regard for anything else
'''
from .asistplanningagents import *

'''
Bias to the top left
'''
class PreEmptiveAgentLeft(PlanAgent):
    def _init_subagent(self, obs):
        pass

    def _get_preferences(self):
        return ['goal', 'explore']


'''
Bias to the bottom right
'''
class PreEmptiveAgentRight(PlanAgent):
    def _init_subagent(self, obs):
        pass

    def _get_preferences(self):
        return ['goal', 'explore']

    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I > 0)
        I[idxneg] = I[idxneg] * 0.5
        return np.abs(I)
