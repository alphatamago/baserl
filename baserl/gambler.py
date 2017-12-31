import sys

import matplotlib.pyplot as plt

from baserl.common import *
from baserl.mdp_base import MDPBase

class Gambler(MDPBase):
    def __init__(self, goal, prob_win):
        self.goal_ = goal
        self.prob_win_ = prob_win
        self.states_ = list(range(0, goal+1))
        self.actions_ = list(range(1, goal))


    def states(self):
        # A state is a number for how much money the gambler has
        return self.states_


    def actions(self, state):
        # Since actions is stake and states is owned money,
        # the stake must be no more than actual owned.
        return [action for action in self.actions_ if action <= state and
                (state + action <= self.goal_)]


    def is_terminal(self, state):
        return state == 0 or state == self.goal_


    def transitions(self, state, action):
        reward = 0
        if state + action >= self.goal_:
            reward = 1
        return [((state + action, reward), self.prob_win_),
                ((state - action, 0), 1.0 - self.prob_win_)]

    def gamma(self):
        # Undiscounted
        return 1.0


    def print_value(self, v):
        x_axis = []
        y_axis = []
        for state in self.states_:
            if self.is_terminal(state):
                continue
            x_axis.append(state)
            y_axis.append(v[state])
            # print("state:", state, "value:", v[state])
        plt.figure()
        plt.plot(x_axis, y_axis)
        plt.show()


    def print_policy(self, policy):
        x_axis = []
        y_axis = []
        for k, action_probs in policy.items():
            for va in [a[0] for a in action_probs.items() if a[1] > 0]:
                # print("state:", k, "top actions:",
                #      sorted(action_probs.items(), key=lambda x:x[1],
                #             reverse=True)[:10])
                x_axis.append(k)
                y_axis.append(va)
        plt.figure()
        plt.scatter(x_axis, y_axis, marker='.')
        plt.title('policy - showing all actions')
        plt.show()

        x_axis = []
        y_axis = []
        for x, actions in policy.items():
            x_axis.append(x)
            max_p = None
            max_a = None
            # Find the lowest optimal amount to bet
            for action, prob in sorted(actions.items(), key=lambda x: x[0],
                                       reverse=False):
                if max_p is None or max_p < prob:
                    max_p = prob
                    max_a = action
            y_axis.append(max_a)
        plt.title('policy - showing lowest stake that is an optimal action')
        plt.plot(x_axis, y_axis)
        plt.show()
