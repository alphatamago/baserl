import copy
import math
import random
import sys
import time

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm

TOLERANCE_CMP_VALUES = 0.000001

def make_random_policy(states, actions):
    random_policy = {}
    for state in states:
        random_policy[state] = {}
        possible_actions = actions(state) 
        for action in possible_actions:
            random_policy[state][action] = (1.0 / len(possible_actions))
    return random_policy

def compute_action_value(v, state_action_transitions, gamma):
    sum_pr = 0
    for ((next_state, reward), p_next) in state_action_transitions:
        vnext = 0
        if next_state in v:
            vnext = v[next_state]
        sum_pr += (p_next * (reward + gamma * vnext))
    return sum_pr


def compute_state_value(state, v, policy, transitions, gamma):
    sum_v = 0
    for action, pi_a_s in policy[state].items():
        sum_v += (pi_a_s * compute_action_value(v, transitions(state, action), gamma))
    return sum_v


def make_greeedy_policy_from_v(v, states, actions, transitions, gamma):
    policy = {}
    for state in states:
        assert state not in policy
        policy[state] = {}
        # Find the highest value actions from current state
        max_v = None
        for action in actions(state):
            val = compute_action_value(v, transitions(state, action), gamma)
            assert action not in policy[state]
            policy[state][action] = val
            if max_v is None or max_v <= val:
                max_v = val
        # Find all actions that have the highest value - there can be more than one
        num_best_actions = 0
        # Going over actions once to count how many best actions are there
        for action, val in policy[state].items():
            if abs(val - max_v) < TOLERANCE_CMP_VALUES:
                num_best_actions += 1
        assert num_best_actions > 0
        # Going over actions a 2nd time to share uniform probability among best actions,
        # and zero for the sub-optimal ones
        for action, val in policy[state].items():
            if abs(val - max_v) < TOLERANCE_CMP_VALUES:
                policy[state][action] = 1.0 / num_best_actions
            else:
                policy[state][action] = 0
        
    return policy


def iterative_policy_evaluation(policy, theta, states, is_terminal, actions, transitions, gamma,
                                in_place=True,
                                max_iter=1000,
                                print_value=None,
                                print_policy=None,
                                print_every_n=1,
                                verbose=False):
    """
    This implements the algorithm with the same name from Sutton's Reinforcement Learning book, 2nd edition, page 61
    
    max_iter: not part of the original algorithm: it is meant as a safety-guard to prevent it running for too long
    
    print_value and print_policy: optional functions that can be useful to visualize the progress
    print_every_n: the frequency of calling print_value and print_policy
    """
    v = dict([(s, 0) for s in states])
    num_iter = 0
    delta = theta

    if print_value is not None:
        print("Initial value function:")
        print_value(v, states)
        print()
    if print_policy is not None:
        print("Initial greedy policy:")
        print_policy(make_greeedy_policy_from_v(v, states, actions, transitions, gamma), states, actions)
        print()

    while delta >= theta:
        if num_iter > max_iter:
            print ('Stopped early due to num_iter > max_iter:', num_iter, max_iter)
            break
        delta = 0
        num_iter += 1
        if in_place:
            old_v = v
        else:
            old_v = copy.deepcopy(v)
        for state in states:
            if is_terminal(state): continue
            val = v[state]
            v[state] = compute_state_value(state, old_v, policy, transitions, gamma)
            delta = max(delta, abs(val-v[state]))
        if num_iter % print_every_n == 0:
            if print_value is not None:
                print("value function at iteration", num_iter)
                print_value(v, states)
                print()
            if print_policy is not None:
                print("greedy policy at iteration", num_iter)
                print_policy(make_greeedy_policy_from_v(v, states, actions, transitions, gamma), states, actions)
                print()
        if verbose:
            print("iterative_policy_evaluation: num iter=", num_iter, "delta=", delta)
        
    if print_value is not None:
        print("Final value function after #iters =", num_iter)
        print_value(v, states)
    if print_policy is not None:
        print("Final greedy policy after #iters =", num_iter)
        print_policy(make_greeedy_policy_from_v(v, states, actions, transitions, gamma), states, actions)
    return v

def policy_iteration(states, is_terminal, actions, transitions, gamma, 
                     policy_evaluator,
                     delta_policy_improv, max_iter_policy_improv, 
                     initial_policy=None,
                     print_value=None,
                     print_policy=None):
    """
    Implementing "Policy Iteration using iterative policy evaluation" (page 65 from Sutton's Reinforcement Learning, 2nd ed)
    
    Given the definition of a MDP (states, is_terminal, actions, transition probabilities p, gamma),
    we try to find the best deterministic policy to pick actions in given states.
    policy_evaluator is a function used to compute value functions for a given policy.    
    """
    if initial_policy is None:
        initial_policy = make_random_policy(states, actions)
    current_policy = copy.deepcopy(initial_policy)

    current_v = {}
    is_policy_stable = False
    num_iter = 0
    while not is_policy_stable:
        if num_iter > max_iter_policy_improv:
            break
        num_iter += 1
        
        # evalute the current policy
        new_v = policy_evaluator(
            states=states,
            is_terminal=is_terminal,
            actions=actions,
            transitions=transitions,
            policy=current_policy,
            gamma=gamma)
        
        current_v = new_v

        updated_policy = make_greeedy_policy_from_v(current_v, states, actions, transitions, gamma)
        
        # Compare updated_policy with current_policy
        is_policy_stable = True
        for state in states:
            if not is_policy_stable:
                break
            for action, prob in current_policy[state].items():
                if abs(updated_policy[state][action] - prob) > TOLERANCE_CMP_VALUES:
                    is_policy_stable = False
                    break

        # Update the current policy to the improved one
        current_policy = updated_policy

        if print_value is not None:
                print("value function at iteration", num_iter)
                print_value(current_v, states)
                print()
        if print_policy is not None:
                print("greedy policy at iteration", num_iter)
                print_policy(current_policy, states, actions)
                print()
                
    return current_policy, current_v

def make_iterative_policy_evaluator(theta, max_iter):
    def f(states, is_terminal, actions, transitions, policy, gamma):
        return iterative_policy_evaluation(policy, theta, states, is_terminal, actions, transitions, gamma, max_iter=max_iter)
    return f
