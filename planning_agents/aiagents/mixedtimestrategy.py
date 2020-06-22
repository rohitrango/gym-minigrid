'''
This will contain agents that have the following plan:
    - For some time go for both yellow and green victims
    - Then prioritize the yellow victims
    - then go back to the green ones
'''
from .asistplanningagents import *

class MixedTimeAgent(PlanAgent):
    '''
    This agent will triage yellow ones till time is up and then save green victims
    '''
    def __init__(self, *args, **kwargs):
        self.procrastination_fraction = kwargs.pop('procrastination_frac', 0.8)
        super().__init__(*args, **kwargs)

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
            if self.time <= int(self.procrastination_fraction * self.victimlifetime):
                self.victimcolor = ['yellow', 'green']
                return ['goal', 'explore']
            else:
                self.victimcolor = ['yellow']
                return ['goal yellow', 'explore']
        else:
            self.victimcolor = ['yellow', 'green']
            return ['goal green', 'explore']


class MixedTimeAgentLeft(MixedTimeAgent):
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


class MixedTimeAgentRight(MixedTimeAgent):
    '''
    Bias to the bottom right
    '''
    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I > 0)
        I[idxneg] = I[idxneg] * 0.5

