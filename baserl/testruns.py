import random
import sys
import time

from baserl.common import *
from baserl.gambler import Gambler
from baserl.grid_world import GridWorld
from baserl.jacks_rental import JacksRental


def test_grid_world_policy_iteration():
    grid_world = GridWorld()
    start_time = time.time()
    grid_world_policy, grid_world_v = policy_iteration(
        states=grid_world.states(), 
        is_terminal=grid_world.is_terminal, 
        actions=grid_world.actions,
        transitions=grid_world.transitions,
        gamma=grid_world.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.0001,
                                                         max_iter=150),
        delta_policy_improv=0.00000001,
        max_iter_policy_improv=10,
        print_value=grid_world.print_value,
        print_policy=grid_world.print_policy)
    print("Done in time:", time.time()-start_time)
    grid_world.report_stats()


def test_grid_world_value_iteration():
    grid_world = GridWorld()
    start_time = time.time()
    grid_world_policy, grid_world_v = value_iteration(
        states=grid_world.states(), 
        is_terminal=grid_world.is_terminal, 
        actions=grid_world.actions,
        transitions=grid_world.transitions,
        gamma=grid_world.gamma(),
        delta_threshold=0.00000001,
        max_iter=10,
        print_value=grid_world.print_value,
        print_policy=grid_world.print_policy)    
    print("Done in time:", time.time()-start_time)
    grid_world.report_stats()


def test_jacks_rental_policy_iteration(max_cars, prob_threshold):
    jacks_rental = JacksRental(max_cars=max_cars,
                               transitions_prob_threshold=prob_threshold)
    start_time = time.time()
    jacks_rental_policy, jacks_rental_v = policy_iteration(
        states=jacks_rental.states(), 
        is_terminal=jacks_rental.is_terminal, 
        actions=jacks_rental.actions,
        transitions=jacks_rental.transitions,
        gamma=jacks_rental.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.000001,
                                                         max_iter=200),
        delta_policy_improv=0.000001,
        max_iter_policy_improv=10,
        print_value=None,
        print_policy=jacks_rental.print_policy)
    print("Done in time:", time.time()-start_time)
    jacks_rental.report_stats()


def test_jacks_rental_value_iteration(max_cars, prob_threshold):
    jacks_rental = JacksRental(max_cars=max_cars,
                               transitions_prob_threshold=prob_threshold)
    start_time = time.time()
    jacks_rental_policy, jacks_rental_v = value_iteration(
        states=jacks_rental.states(), 
        is_terminal=jacks_rental.is_terminal, 
        actions=jacks_rental.actions,
        transitions=jacks_rental.transitions,
        gamma=jacks_rental.gamma(),
        delta_threshold=0.1,
        max_iter=100,
        print_value=None,
        print_policy=jacks_rental.print_policy)
    print("Done in time:", time.time()-start_time)
    jacks_rental.report_stats()
    

def test_jacks_rental_policy_iteration(max_cars, prob_threshold):
    jacks_rental = JacksRental(max_cars=max_cars,
                               transitions_prob_threshold=prob_threshold)
    start_time = time.time()
    jacks_rental_policy, jacks_rental_v = policy_iteration(
        states=jacks_rental.states(), 
        is_terminal=jacks_rental.is_terminal, 
        actions=jacks_rental.actions,
        transitions=jacks_rental.transitions,
        gamma=jacks_rental.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.000001,
                                                         max_iter=200),
        delta_policy_improv=0.000001,
        max_iter_policy_improv=10,
        print_value=None,
        print_policy=jacks_rental.print_policy)
    print("Done in time:", time.time()-start_time)
    jacks_rental.report_stats()


def test_gambler_policy_iteration():
    mdp = Gambler(goal=100, prob_win=0.25)
    start_time = time.time()
    mdp_policy, mdp_v = policy_iteration(
        states=mdp.states(), 
        is_terminal=mdp.is_terminal, 
        actions=mdp.actions,
        transitions=mdp.transitions,
        gamma=mdp.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.000001,
                                                         max_iter=200),
        delta_policy_improv=0.000001,
        max_iter_policy_improv=10,
        print_value=mdp.print_value,
        print_policy=mdp.print_policy)
    print("Done in time:", time.time()-start_time)
    mdp.report_stats()

    
def test_gambler_value_iteration():
    mdp = Gambler(goal=100, prob_win=0.25)
    start_time = time.time()
    mdp_policy, mdp_v = value_iteration(
        states=mdp.states(), 
        is_terminal=mdp.is_terminal, 
        actions=mdp.actions,
        transitions=mdp.transitions,
        gamma=mdp.gamma(),
        delta_threshold=0.00000001,
        max_iter=30,
        print_value=mdp.print_value,
        print_policy=mdp.print_policy)
    print("Done in time:", time.time()-start_time)
    mdp.report_stats()


if __name__ ==  "__main__":
    random.seed(42)

    print("Running policy iteration for Grid World")
    test_grid_world_policy_iteration()
    print()

    print("Running value iteration for Grid World")    
    test_grid_world_value_iteration()
    print()

    jack_max_cars = 10
    jack_prob_threshold = 20
    print("Running policy iteration for Jack's Rental")
    test_jacks_rental_policy_iteration(jack_max_cars, jack_prob_threshold)
    print()

    print("Running value iteration for Jack's Rental")    
    test_jacks_rental_value_iteration(jack_max_cars, jack_prob_threshold)
    print()

    print("Running policy iteration for Gambler's Problem")    
    test_gambler_policy_iteration()
    print()

    print("Running value iteration for Gambler's Problem")
    test_gambler_value_iteration()
    print()
