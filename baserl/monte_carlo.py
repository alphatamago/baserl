import copy
import random

from collections import defaultdict

from baserl.common import random_sample

def monte_carlo_policy_evaluation(every_visit,
                                  policy,
                                  gamma,
                                  episode_generator,
                                  num_episodes,
                                  verbose=False,
                                  v_history=None):
    """
    Implements both "First-visit" and "Every-visit" versions of Monte Carlo
    prediction for the value function of a given policy.
    The implementation follows the algorithm as described in Sutton's Reinforcement
    Learning book, 2nd edition, page 76.

    @every_visit: if True, runs the "Every-visit" version, else runs "First-visit".

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
        if v_history is not None:
            v_history.append(copy.deepcopy(v))
    return v


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
