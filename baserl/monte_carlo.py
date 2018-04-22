import copy
import random

from collections import defaultdict

from baserl.common import *

def select_random_legal_action(policy, state):
    sum_p = 0
    legal_actions = []
    for a, p_a in policy[state].items():
        if p_a < 0:
            print (state, policy[state].items())
        assert p_a >= 0
        if p_a > 0:
            sum_p += p_a
            legal_actions.append((a, p_a))
    return random_sample(legal_actions)


def monte_carlo_policy_evaluation(every_visit,
                                  policy,
                                  gamma,
                                  episode_generator,
                                  num_episodes,
                                  verbose=False,
                                  v_history=None,
                                  v_history_snapshots=None):
    """
    Implements both "First-visit" and "Every-visit" versions of Monte Carlo
    prediction for the value function of a given policy.
    The implementation follows the algorithm as described in Sutton's
    Reinforcement Learning book, 2nd edition, page 76.

    @every_visit: if True, runs the "Every-visit" version, else runs
    "First-visit".

    The algorithm doesn't have any specific termination condition, therefore
    here we use num_episodes to control the run time.
    """

    v = defaultdict(float)

    # episode_returns[s] = (sum_returns, num_returns)
    episode_returns = defaultdict(lambda: (0,0))

    # We don't count the empty episodes towards num_episodes
    num_non_empty_episodes = 0

    while num_non_empty_episodes < num_episodes:
        episode = episode_generator.generate(policy)
        if verbose:
            print("episode:", episode)
        # returns_in_episode[state] = [(sum_rewards, gamma_pow)]
        returns_in_episode = defaultdict(list)
        for (state, action, reward) in episode:
            if every_visit or state not in returns_in_episode: 
                returns_in_episode[state].append((0, 0))
            for k, stats_list in returns_in_episode.items():
                for i in range(len(stats_list)):
                    (sum_so_far, gamma_pow) = stats_list[i]
                    stats_list[i] = (sum_so_far + (gamma**gamma_pow) * reward,
                                     gamma_pow + 1)

        if verbose:
            print ("returns_in_episode:", returns_in_episode)
        if len(returns_in_episode) == 0:
            # Empty episode, skip
            continue
        num_non_empty_episodes += 1
        for state, returns_list in returns_in_episode.items():
            for returns in returns_list:
                (sum_vals, count_vals) = episode_returns[state]
                updated_sum = sum_vals + returns[0]
                updated_count = count_vals + 1
                episode_returns[state] = (updated_sum, updated_count)
                v[state] = 1.0 * updated_sum / updated_count
        if ((v_history is not None) and
            (v_history_snapshots is None or
             (num_non_empty_episodes in v_history_snapshots))):
            v_history.append(copy.deepcopy(v))
    return v


class ModelBasedEpisodeGenerator:
    """
    This is an episode generator that is based on a fully defined MDP
    environment.
    This is convenient when comparing Monte-Carlo methods with DP methods, for
    environments where the transition distribution is defined. These are not best
    use-cases for Monte-Carlo methods in practice, by the way, but are good for
    experimentation.
    """
    def __init__(self, states, is_terminal, actions, transitions,
                 max_episode_len,
                 fixed_start_state,
                 with_exploring_starts,
                 verbose=False):
        self.states_ = states
        self.is_terminal_ = is_terminal
        self.actions_ = actions
        self.non_terminal_states_ = [s for s in self.states_
                                     if not self.is_terminal_(s)]
        assert len(self.non_terminal_states_) > 0 
        self.transitions_ = transitions
        self.max_episode_len_ = max_episode_len
        self.verbose_ = verbose
        self.fixed_start_state_ = fixed_start_state
        self.with_exploring_starts_ = with_exploring_starts


    def generate(self, policy, start_state=None, start_action=None):
        assert ((start_action is None) or (not self.with_exploring_starts_))
        if self.fixed_start_state_ is not None:
            start_state = self.fixed_start_state_
        if self.verbose_:
            print ("generate episode max_episode_len:", self.max_episode_len_,
                   "start_state:", start_state,
                   "start_action:, start_action")
        # Sample a starting state
        if start_state is None:
            start_state = self.non_terminal_states_[
                random.randint(0, len(self.non_terminal_states_)-1)]
        current_s = start_state
        episode = []
        for i in range(self.max_episode_len_):
            if self.is_terminal_(current_s):
                if self.verbose_:
                    print("Finishing early, terminal state", current_s)
                break
            if i == 0: 
                if start_action is not None:
                    selected_action = start_action
                else:
                    if self.with_exploring_starts_:
                        selected_action = self.actions_[
                            random.randint(0, len(self.actions_)-1)]
                    else:
                        selected_action = select_random_legal_action(policy,
                                                                     current_s)
            else:
                selected_action = select_random_legal_action(policy, current_s)
                
            # Transition to next state, according to probability transitions p
            (new_s, reward) = random_sample(self.transitions_(current_s,
                                                              selected_action))
            episode.append((current_s, selected_action, reward))
            current_s = new_s
        return episode


def on_policy_monte_carlo_control(initial_policy, gamma, 
                                  episode_generator,
                                  num_episodes,
                                  epsilon=0):
    """
    This function can be used either to run "Monte Carlo ES (Exploring Starts" as
    described in Sutton's RL book, 2nd ed, page 81, as well as to run "On Policy
    first-visit MC control (for epsilon-soft policies)" as described on page 83.

    To run the Exploring Starts version, set epsilon to zero (the default value)
    and provide an episode_generator that has the exploring-starts property.

    To run the Epsilon-Soft Policies version, one can use any episode_generator
    but make sure to set epsilon to a non-zero value.
    """
    assert num_episodes > 0
    current_policy = copy.deepcopy(initial_policy)

    # q[state][action] = (sum_returns, num_returns) - in order to compute average
    # returns from taking given action in given state
    q = {}
    num_non_empty_episodes = 0
    num_empty_episodes = 0
    measured_initial_policy = None
    while num_non_empty_episodes < num_episodes:
        episode_states = set()
        episode = episode_generator.generate(current_policy)

        # The reason for keeping track of gamma_power separately per (s,a) pair is
        # to discount separately from each first encounter of a (s,a) pair in an
        # episode. A particular (s,a) can occur first time in the middle of the
        # episode, then we start it's Q calculation with full reward, then next
        # reward gets discounted by gamma, the one after by gamma^2, etc
        # episode_returns[(s, a)] = sum_rewards, gamma_power
        episode_returns = {}
        for (state, action, reward) in episode:
            if state not in current_policy:
                assert(False)
            if action not in current_policy[state]:
                assert(False)                
            episode_states.add(state)
            if (state, action) not in episode_returns:
                episode_returns[(state, action)] = (0, 0)
            for (s, a) in episode_returns:
                (sum_rewards, gamma_power) = episode_returns[(s, a)]
                episode_returns[(s, a)] =  (sum_rewards  + reward *
                                            (gamma**gamma_power),
                                            gamma_power + 1)
                
        if len(episode_states) == 0:
            # This is needed for instance in blackjack - not sure what to do with
            # the episodes that start in a terminal state, where it ends with no
            # action
            num_empty_episodes += 1
            continue
            
        num_non_empty_episodes += 1

        # Update current Q-values
        for (state, action) in episode_returns:
            if state not in q:
                q[state] = {}
            if action not in q[state]:
                q[state][action] = (0, 0)
            sum_values, num_values = q[state][action]
            q[state][action] = (sum_values + episode_returns[(state, action)][0],
                                num_values + 1)

        # Update current policy
        assert (len(episode_states) > 0)
        for s in episode_states:
            # Find the highest value
            max_v = None
            for a, (sum_vals, num_vals) in q[s].items():
                v = 1.0 * sum_vals / num_vals
                if max_v is None or max_v <= v:
                    max_v = v
            
            # Find all actions that have the highest value - there can be more
            # than one
            max_value_actions = set()
            for a, (sum_vals, num_vals) in q[s].items():
                v = 1.0 * sum_vals / num_vals
                # Since we use floating points, we use tolerance comparisons
                if abs(v - max_v) < TOLERANCE_CMP_VALUES:
                    max_value_actions.add(a)
                    
            assert len(max_value_actions) > 0

            # Assign values to the actions
            possible_actions = current_policy[s].keys()
            assert len(possible_actions) > 0
            sum_probs = 0
            for a in possible_actions:
                if a in max_value_actions:
                    prob = ((1.0 - epsilon)/len(max_value_actions) +
                            epsilon/len(possible_actions))
                    current_policy[s][a] = prob
                    sum_probs += prob
                else:
                    prob = epsilon/len(possible_actions)
                    current_policy[s][a] = prob
                    sum_probs += prob
            # Check the probability distribution assumption
            assert abs(sum_probs - 1.0) < TOLERANCE_CMP_VALUES

    return current_policy, q


def off_policy_monte_carlo_policy_evaluation(target_policy,
                                             behavior_policy,
                                             weighted_importance_sampling,
                                             gamma, 
                                             episode_generator,
                                             num_episodes,
                                             start_state=None,
                                             q_history=None):
    """
    Estimates state-action pairs Q function for a given target_policy, by only
    observing episodes generated by a separate, behavior_policy, using off-policy
    importance-sampling for monte-carlo.

    If weighted_importance_sampling is True, it does what it says for averaging,
    otherwise it uses "ordinary importance sampling" averaging.

    The implementation follows the "Off-policy MC prediction (policy evaluation)
    for estimating Q ~ q_pi" algorithm as described in Sutton's
    Reinforcement Learning book, 2nd edition, page 109.

    The algorithm doesn't have any specific termination condition, therefore
    here we use num_episodes to control the run time.
    """

    assert num_episodes > 0

    # q[state][action] = average returns from taking given action in given state
    q = {}
    # Same as q, used for incremental calculations of averages.
    C = {}

    num_non_empty_episodes = 0
    measured_initial_policy = None
    a_max = None
    while num_non_empty_episodes < num_episodes:
        episode_states = set()
        # Generate an episode using the behavior policy
        episode = episode_generator.generate(behavior_policy, start_state)
        if len(episode) > 0:
            num_non_empty_episodes += 1
        else:
            continue
        G = 0
        W = 1
        episode.reverse()
        for (state, action, reward) in episode:
            if state not in behavior_policy:
                assert(False)
            if action not in behavior_policy[state]:
                assert(False)
            G *= gamma
            G += reward

            if state not in C:
                C[state] = {}
            if action not in C[state]:
                C[state][action] = 0

            if weighted_importance_sampling:
                C[state][action] += W
            else:
                C[state][action] += 1

            if state not in q:
                q[state] = {}
            if action not in q[state]:
                q[state][action] = 0

            q_delta = W * (G - q[state][action]) / C[state][action]
            q[state][action] += q_delta

            W = W * target_policy[state][action] / behavior_policy[state][action]
            if W == 0:
                break
        if q_history is not None and start_state is not None:
            if a_max is None:
                v_max = None
                for a, v in target_policy[start_state].items():
                    if v_max is None or v_max < v:
                        v_max = v
                        a_max = a
            q_val = 0
            try:
                q_val = q[start_state][a_max]
            except KeyError:
                pass
            q_history.append(q_val)
    return q
