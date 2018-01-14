import random
import sys
import time

from baserl.blackjack import *
from baserl.common import *
from baserl.gambler import Gambler
from baserl.grid_world import GridWorld
from baserl.jacks_rental import JacksRental


def test_grid_world_policy_iteration():
    print("Running policy iteration for Grid World")
    grid_world = GridWorld()
    start_time = time.time()
    grid_world_policy, grid_world_v = policy_iteration(
        states=grid_world.states(), 
        is_terminal=grid_world.is_terminal, 
        actions=grid_world.actions,
        transitions=grid_world.transitions,
        gamma=grid_world.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.0001,
                                                         max_iter=50),
        delta_policy_improv=0.0001,
        max_iter_policy_improv=10,
        print_value=grid_world.print_value,
        print_policy=grid_world.print_policy)
    print("Done in time:", time.time()-start_time)
    grid_world.report_stats()


def test_grid_world_value_iteration():
    print("Running value iteration for Grid World")
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
                                                         max_iter=50),
        delta_policy_improv=0.0001,
        max_iter_policy_improv=10,
        print_value=None,
        print_policy=jacks_rental.print_policy)
    print("Done in time:", time.time()-start_time)
    jacks_rental.report_stats()


def test_jacks_rental_value_iteration(max_cars, prob_threshold):
    print("Run Value Iteration for Jack's Rental Problem")    
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


def test_gambler_policy_iteration():
    print("Run Policy Iteration for Gambler's Problem")    
    mdp = Gambler(goal=100, prob_win=0.25)
    start_time = time.time()
    mdp_policy, mdp_v = policy_iteration(
        states=mdp.states(), 
        is_terminal=mdp.is_terminal, 
        actions=mdp.actions,
        transitions=mdp.transitions,
        gamma=mdp.gamma(),
        policy_evaluator=make_iterative_policy_evaluator(theta=0.000001,
                                                         max_iter=50),
        delta_policy_improv=0.0001,
        max_iter_policy_improv=10)
    print("Done in time:", time.time()-start_time)

    
def test_gambler_value_iteration():
    print("Run Value Iteration for Gambler's Problem")
    mdp = Gambler(goal=100, prob_win=0.25)
    start_time = time.time()
    mdp_policy, mdp_v = value_iteration(
        states=mdp.states(), 
        is_terminal=mdp.is_terminal, 
        actions=mdp.actions,
        transitions=mdp.transitions,
        gamma=mdp.gamma(),
        delta_threshold=0.00000001,
        max_iter=30)
    print("Done in time:", time.time()-start_time)


def test_blackjack_monte_carlo_first_visit_policy_evaluation():
    print("Evaluate simple policy for Blackjack using Monte-Carlo First-Visit")
    mdp = Blackjack()
    episode_generator = BlackjackEpisodeGenerator(with_exploring_starts=False)
    simple_policy = mdp.make_simple_blackjack_player_policy()
    start_time = time.time()
    v_history = []
    v = monte_carlo_policy_evaluation(every_visit=False,
                                      policy=simple_policy,
                                      gamma=mdp.gamma(),
                                      episode_generator=episode_generator,
                                      num_episodes=100000,
                                      v_history=v_history)
    print("Done in time:", time.time()-start_time)    


def test_blackjack_monte_carlo_control_exploring_starts():
    print("Generate a better Blackjack policy with Monte Carlo Exploring Starts")
    mdp = Blackjack()
    initial_policy = mdp.make_simple_blackjack_player_policy() 
    exploring_starts_episode_generator = BlackjackEpisodeGenerator(
        with_exploring_starts=True)
    start_time = time.time()
    final_policy, q = on_policy_monte_carlo_control(
        initial_policy=initial_policy,
        gamma=mdp.gamma(), 
        episode_generator=exploring_starts_episode_generator,
        num_episodes=10000,
        epsilon=0)
    print("Done in time:", time.time() - start_time)    


def test_blackjack_monte_carlo_control_epsilon_soft():
    print("Generate a better Blackjack policy with Monte Carlo Epsilon-Soft")
    mdp = Blackjack()
    initial_policy = mdp.make_simple_blackjack_player_policy() 
    non_exploring_starts_episode_generator = BlackjackEpisodeGenerator(
        with_exploring_starts=False)
    start_time = time.time()
    final_policy, q = on_policy_monte_carlo_control(
        initial_policy=initial_policy,
        gamma=mdp.gamma(), 
        episode_generator=non_exploring_starts_episode_generator,
        num_episodes=10000,
        epsilon=0.1)
    print("Done in time:", time.time() - start_time)    


if __name__ ==  "__main__":
    random.seed(42)

    test_blackjack_monte_carlo_first_visit_policy_evaluation()
    print()
    
    test_grid_world_policy_iteration()
    print()

    test_grid_world_value_iteration()
    print()

    jack_max_cars = 10
    jack_prob_threshold = 20
    test_jacks_rental_policy_iteration(jack_max_cars, jack_prob_threshold)
    print()

    test_jacks_rental_value_iteration(jack_max_cars, jack_prob_threshold)
    print()

    test_gambler_policy_iteration()
    print()

    test_gambler_value_iteration()
    print()

    test_blackjack_monte_carlo_control_exploring_starts()
    print()

    test_blackjack_monte_carlo_control_epsilon_soft()
    print()
