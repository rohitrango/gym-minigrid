'''
This will contain agents that have the following plan:
    - Prioritize the yellow victims and save green if they have a small threshold
    - then go back to the green ones
'''
from .asistplanningagents import *

class MixedProximityAgent(PlanAgent):
    '''
    This agent will triage yellow ones and if green victims are nearby, then also save green ones
    '''
    def __init__(self, *args, **kwargs):
        self.astardelta = kwargs.pop('astardelta', 0)
        super().__init__(*args,)

    def _init_subagent(self, obs):
        self.time = obs['time']
        self.victimlifetime = obs['victimlifetime']
        self.numyellow = obs['numyellow']
        self.numgreen = obs['numgreen']
        self.reward_green = obs['reward_green']
        self.reward_yellow = obs['reward_yellow']

        # Only save yellow victims now
        self.victimcolor = ['yellow']
        self.last_triaged = 0


    def _get_preferences(self):
        # This is where we should focus on yellow ones
        if self.time <= self.victimlifetime and self.numyellow > 0:
            green_victim_nearby = False
            # TODO: Check for nearby green victims
            if self.astardelta > 0 and self.last_triaged > 0:
                # Get belief
                A = self.agent_view_size
                belief = self.belief[A:-A, A:-A] + 0
                prob = belief[:, :, OBJECT_TO_IDX['goal'] - 1] * self.colors[A:-A, A:-A, COLOR_TO_IDX['green']]
                x, y = np.where(prob > 0.7)
                px, py = self.agent_pos
                for _x, _y in zip(x, y):
                    if np.abs(px - _x) + np.abs(py - _y) <= self.astardelta:
                        path = self.astar([_x, _y])
                        path = list(filter(lambda x: x == 'forward', path))
                        if len(path) < self.astardelta:
                            green_victim_nearby = True
                            break

            # If there are green victims nearby, then keep all prefs
            if green_victim_nearby:
                self.victimcolor = ['yellow', 'green']
                return ['goal', 'explore']
            else:
                self.victimcolor = ['yellow']
                return ['goal yellow', 'explore']
        else:
            self.victimcolor = ['yellow', 'green']
            return ['goal green', 'explore']


    def _update_from_reward(self, rew):
        if self.last_triaged > 0:
            self.last_triaged -= 1

        if rew is None or self.reward_green is None or self.reward_yellow is None:
            return None

        if rew == self.reward_green:
            self.numgreen -= 1
        elif rew == self.reward_yellow:
            self.numyellow -= 1
            self.last_triaged = self.astardelta + 5



class MixedProximityAgentLeft(MixedProximityAgent):
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


class MixedProximityAgentRight(MixedProximityAgent):
    '''
    Bias to the bottom right
    '''
    def leakyabs(self, I, d):
        '''
        Use this to create bias in your agent
        '''
        idxneg = (I > 0)
        I[idxneg] = I[idxneg] * 0.5

