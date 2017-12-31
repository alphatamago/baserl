import sys

from baserl.common import *
from baserl.mdp_base import MDPBase

RIGHT = 'R'
LEFT = 'L'
UP = 'U'
DOWN = 'D'

class GridWorld(MDPBase):
    """
    This is a 2-D environemnt, with moves between adjacent cells horizontally
    or vertically.
    When attempted to move outside, nothing happens.
    Each transition costs 1 unit (or -1 reward). When any of the terminal
    states is reached, the episode stops - so the goal is to reach a terminal
    state as soon as possible, to minimize the losses.

    This is modeled as an undiscounted episodic MDP.
    """

    def __init__(self):
        self.grid_ = [
            ['T', ' ', ' ', ' '],
            [' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' '],
            [' ', ' ', ' ', 'T']]


    def states(self):
        return  [(x, y) for x in range(len(self.grid_))
                 for y in range(len(self.grid_[0]))]


    def actions(self, state):
        return (UP, DOWN, RIGHT, LEFT)


    def is_terminal(self, state):
        return self.grid_[state[0]][state[1]] == 'T'


    def gamma(self):
        # Undiscounted
        return 1.0


    def transitions(self, state, action):
        """
        Here, the transitions are deterministic, so we always return a point
        distribution.
        """
        if action == RIGHT:
            if state[1] < (len(self.grid_[0]) - 1):
                return [(((state[0], state[1] + 1), -1), 1.0)]
            else:
                # stay in the same state
                return [((state,  -1), 1.0)]
        if action == LEFT:
            if state[1] > 0:
                return [(((state[0], state[1] - 1), -1), 1.0)]
            else:
                # stay in the same state
                return [((state,  -1), 1.0)]
        if action == DOWN:
            if state[0] < (len(self.grid_)-1):
                return [(((state[0] + 1, state[1]), -1), 1.0)]
            else:
                # stay in the same state
                return [((state,  -1), 1.0)]
        if action == UP:
            if state[0] > 0:
                return [(((state[0] - 1, state[1]), -1), 1.0)]
            else:
                # stay in the same state
                return [((state, -1), 1.0)]


    def print_value(self, v):
        for x in range(len(self.grid_)):
            for y in range(len(self.grid_[0])):
                sys.stdout.write("%.2f " % v[(x, y)])
            sys.stdout.write("\n")


    def print_policy(self, policy):
        for x in range(len(self.grid_)):
            for y in range(len(self.grid_[0])):
                # Find best actions from state (x, y)
                max_prob = None
                for action, prob in policy[(x, y)].items():
                    if max_prob is None or max_prob < prob:
                        max_prob = prob
                # Looping 2nd time to collect best actions, now that we now the
                # max prob
                best_actions = []
                for action, prob in policy[(x, y)].items():
                    if abs(prob - max_prob) < TOLERANCE_CMP_VALUES:
                        best_actions.append(action)
                sys.stdout.write("%4s " % ''.join(best_actions))
            sys.stdout.write("\n")
