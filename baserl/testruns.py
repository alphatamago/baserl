import random
import sys
import time

from baserl.common import *
from baserl.grid_world import GridWorld
from baserl.jacks_rental import JacksRental

if __name__ ==  "__main__":
    random.seed(42)

    """
eval: num iter= 68
eval: num iter= 69
eval: num iter= 70
Number of states where policy changed: 0
greedy policy at iteration 2
 0 -1 -2 -3 -4 -5
 0 -1 -2 -3 -4 -5
 0 -1 -2 -3 -4 -4
 0 -1 -2 -3 -3 -3
 0 -1 -2 -2 -2 -2
 0 -1 -1 -1 -1 -1

Done in time: 241.51826000213623
    """ 

    """
eval: num iter= 86
eval: num iter= 87
eval: num iter= 88
eval: num iter= 89
Number of states where policy changed: 0
greedy policy at iteration 3
 0  0  0  0  0  0 -1 -1 -2 -2
 1  1  1  1  1  0  0 -1 -1 -1
 2  2  2  2  1  1  0  0  0 -1
 3  3  3  2  2  1  1  1  0  0
 4  4  3  3  2  2  2  1  1  0
 5  4  4  3  3  3  2  2  1  1
 5  5  4  4  4  3  3  2  2  1
 5  5  5  5  4  4  3  3  2  1
 5  5  5  5  5  4  4  3  2  1
 5  5  5  5  5  5  4  3  2  1

Done in time: 695.0891075134277
global_transitions_hits: 273152
global_transitions_misses: 121
    """

    """
eval: num iter= 86
eval: num iter= 87
eval: num iter= 88
Number of states where policy changed: 0
greedy policy at iteration 3
 0  0  0  0  0 -1 -1 -2 -2 -2 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3
 1  1  1  1  0  0 -1 -1 -1 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2 -2
 2  2  2  1  1  0  0  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2
 3  3  2  2  1  1  1  0  0  0  0  0  0  0  0  0  0  0 -1 -1
 4  3  3  2  2  2  1  1  1  1  1  1  1  1  1  1  1  0  0  0
 4  4  3  3  3  2  2  2  2  2  2  2  2  2  2  2  1  1  1  1
 5  4  4  4  3  3  3  3  3  3  3  3  3  3  3  2  2  2  2  1
 5  5  5  4  4  4  4  4  4  4  4  4  4  4  3  3  3  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  4  4  4  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  4  4  4  3  3  3  3  3  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  4  4  4  4  4  4  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1
 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  4  3  2  1

Done in time: 2432.886586666107
global_transitions_hits: 1143450
global_transitions_misses: 441
    """

    """
    From notebook:
    iterative_policy_evaluation: num iter= 23
    iterative_policy_evaluation: num iter= 24
    value function at iteration 3

greedy policy at iteration 3
 0  0 -1 -1 -2 -2 -3 -3 -3 -4 
 1  0  0 -1 -1 -2 -2 -2 -3 -3 
 1  1  0  0 -1 -1 -1 -2 -2 -3 
 2  1  1  0  0  0 -1 -1 -2 -2 
 2  2  1  1  0  0  0 -1 -1 -2 
 3  2  2  1  1  0  0  0 -1 -1 
 3  3  2  2  1  1  1  0  0  0 
 4  3  3  2  2  2  1  1  0  0 
 4  4  3  3  3  2  2  1  1  0 
 5  4  4  4  3  3  2  2  1  1 

Done in time: 69.75549364089966
    """
    
    jacks_rental = JacksRental(max_cars=20)

    # reward_evaluator=jacks_rental.expected_reward_evaluator
    reward_evaluator = make_transitions_based_reward_evaluator(jacks_rental.transitions)
    
    start_time = time.time()
    jacks_rental_policy, jacks_rental_v = policy_iteration(
        states=jacks_rental.states(), 
        is_terminal=jacks_rental.is_terminal, 
        actions=jacks_rental.actions,
        transitions=jacks_rental.transitions,
        gamma=jacks_rental.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.000001,
                                                         max_iter=100,
                                                         reward_evaluator=reward_evaluator,
                                                         verbose=False),
        reward_evaluator=reward_evaluator,
        delta_policy_improv=0.000001,
        max_iter_policy_improv=10,
        print_value=None,
        print_policy=jacks_rental.print_policy,
        verbose=False)
    print("Done in time:", time.time()-start_time)
    jacks_rental.report_stats()
