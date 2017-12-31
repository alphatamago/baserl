import sys

from baserl.common import *

class MDPBase:
    """
    Base class for Markov Decision Process environments.
    """

    def states(self):
        """
        Returns all the possible states.
        """
        raise NotImplementedException()


    def actions(self, state):
        """
        Lists the actions one can take from a given state.
        """
        raise NotImplementedException()        


    def is_terminal(self, state):
        """
        Tells whether a state is terminal or not.
        """
        raise NotImplementedException()
    

    def gamma(self):
        """
        Return the discount value.
        0 for full discount (only current state matters)
        1 for no discount (all rewards matter equally)
        """
        raise NotImplementedException()
    

    def transitions(self, state, action):
        """
        Describes the probability distribution for transitions from a given
        state, if we take a certain action, to new states, including the
        reward in each case (possibly zero), and the probability of that
        particular transition.
        """
        raise NotImplementedException()


    def print_value(self, v):
        raise NotImplementedException()

            
    def print_policy(self, policy):
        raise NotImplementedException()


    def report_stats(self):
        pass
