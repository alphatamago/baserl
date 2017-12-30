import random
import sys
import time

from baserl.common import *
from baserl.grid_world import GridWorld
from baserl.jacks_rental import JacksRental

def test_jacks_rental():
    jacks_rental = JacksRental(max_cars=10,
                               transitions_prob_threshold=10)

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


def test_grid_world():
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

    
if __name__ ==  "__main__":
    random.seed(42)
    test_grid_world()
    test_jacks_rental()
    
