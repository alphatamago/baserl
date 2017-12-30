import sys

from baserl.common import *
from baserl.graphs import heatmap_value_function

UPPER_BOUND_POISSON = 10

MIN_PROB = 1e-20

class JacksRental:
    def __init__(self, 
                 max_cars=10, 
                 max_moves=5,
                 rental_revenue_per_car=10,
                 moving_cost_per_car=2,
                 expected_daily_rentals=[3, 4],
                 expected_daily_returns = [3, 2]):
        self.poisson_cache_ = {}
        # The maximum number of cars at each location
        self.max_cars_ = max_cars
        
        # How many cars can be moved overnight between the two locations
        self.max_moves_ = max_moves
        
        self.rental_revenue_per_car_ = rental_revenue_per_car
        
        self.moving_cost_per_car_ = moving_cost_per_car
        
        # The expected number of daily rentals at each of the two locations
        self.expected_daily_rentals_ = expected_daily_rentals[:]
        
        # The expected number of daily rentals at each of the two locations
        self.expected_daily_returns_ = expected_daily_returns[:]
        
        # Caching the transitions information, to avoid expensive, repeated computations
        self.global_transitions_ = {}
        
        self.precomputed_transitions_per_location_ = None

        self.global_transitions_hits_ = 0
        self.global_transitions_misses_ = 0

    def report_stats(self):
        print("global_transitions_hits:", self.global_transitions_hits_)
        print("global_transitions_misses:", self.global_transitions_misses_)
        

    def states(self):
        """
        Returns all the possible states.
        """
        return   [(x, y) for x in range(0, self.max_cars_ + 1) for y in range(0, self.max_cars_ + 1)]


    def actions(self, state):
        """
        What actions can one take from a given state.
        """
        return range(-min(state[1], self.max_moves_), min(state[0], self.max_moves_) + 1)


    def is_terminal(self, state):
        # There is no terminal state in this problem
        return False
    

    def gamma(self):
        return 0.9
    

    def transitions(self, evening_state, action):
        """
        (state, action) -> [(new_state, reward, prob)]
        state = (num_0, num_1) = counts at the two locations at the end of day
        action = car moved from 0 to 1 (or 1 to 0, if negative)
        reward = revenue from rentals the next day, minus fees for incurred by moving cars (etc)
        new_state = counts at the end of next day, taking into account the overnight moves, rentals and returns next day
        """

        if self.precomputed_transitions_per_location_ is None:
            print("populate_precomputed_transitions_per_location started...")
            start_time = time.time()
            self.populate_precomputed_transitions_per_location_() 
            print("populate_precomputed_transitions_per_location done in time:", time.time() - start_time)
            
        assert type(evening_state) == tuple and len(evening_state) == 2
        assert evening_state[0] >= 0 and evening_state[0] <= self.max_cars_ and evening_state[1] >= 0 and evening_state[1] <= self.max_cars_
        assert action >= -self.max_moves_ and self.max_moves_ <= self.max_moves_
        
        assert len(self.precomputed_transitions_per_location_[0]) > 0
        assert len(self.precomputed_transitions_per_location_[1]) > 0

        state, _ = self.move_cars_(evening_state, action)

        if state in self.global_transitions_:
            self.global_transitions_hits_ += 1
            return list(self.global_transitions_[state].items())
        self.global_transitions_misses_ += 1
        self.global_transitions_[state] = {}

        for (num_rentals_0, num_returns_0, prob_0) in self.precomputed_transitions_per_location_[0]:
            assert prob_0 > 0
            num_rentals_0 = min(num_rentals_0, state[0])
            for (num_rentals_1, num_returns_1, prob_1) in self.precomputed_transitions_per_location_[1]:
                assert prob_1 > 0
                num_rentals_1 = min(num_rentals_1, state[1])
                assert (num_rentals_0 >= 0 and num_rentals_0 <= self.max_cars_ and num_rentals_1 >= 0 and 
                        num_rentals_1 <= self.max_cars_)
                assert (num_returns_0 >= 0 and num_returns_0 <= self.max_cars_ and num_returns_1 >= 0 and 
                        num_returns_1 <= self.max_cars_)
                reward = (num_rentals_0 + num_rentals_1) * self.rental_revenue_per_car_ - abs(action) * self.moving_cost_per_car_

                outcome = (((self.adjust_state_(state[0], num_rentals_0, num_returns_0),
                         self.adjust_state_(state[1], num_rentals_1, num_returns_1)), reward), prob_0*prob_1)
                if outcome[0] not in self.global_transitions_[state]:
                    self.global_transitions_[state][outcome[0]] = 0
                self.global_transitions_[state][outcome[0]] += outcome[1]

        result = list(self.global_transitions_[state].items())
        sum_vals = sum([v for _, v in result])
        if abs(sum_vals-1.0) >= 0.01:
            print ('expected sum probs 1, got:', sum_vals, 'state:', state, 'action:', action)
            assert(False)
        return result


    def print_value(self, v, states):
        # If one wants to print out the values: heatmap_value_function(v, print_format="%3.2f ")
        heatmap_value_function(v)

            
    def print_policy(self, policy, states, actions):
        for x in range(self.max_cars_):
            for y in range(self.max_cars_):
                max_prob = None
                best_action = None
                for action, prob in policy[(x, y)].items():
                    if max_prob is None or max_prob < prob:
                        max_prob = prob
                        best_action = action
                sys.stdout.write("%2s " % best_action)
            sys.stdout.write("\n")


    def expected_reward_evaluator(self, v, evening_state, action, gamma):
        assert(False)
        state, _ = self.move_cars_(evening_state, action)

        expected_reward = 0
        for num_rentals_0 in range(state[0] + 1):
            p_rental_0 = self.poisson_pmf_(mu=self.expected_daily_rentals_[0], k=num_rentals_0)
            if p_rental_0 < MIN_PROB:
                continue
            for num_returns_0 in range(UPPER_BOUND_POISSON + 1):
                p_return_0 = self.poisson_pmf_(mu=self.expected_daily_returns_[0], k=num_returns_0)
                if p_return_0 < MIN_PROB:
                    continue
                for num_rentals_1 in range(state[0] + 1):
                    p_rental_1 = self.poisson_pmf_(mu=self.expected_daily_rentals_[1], k=num_rentals_1)
                    if p_rental_1 < MIN_PROB:
                        continue
                    for num_returns_1 in range(UPPER_BOUND_POISSON + 1):
                        p_return_1 = self.poisson_pmf_(mu=self.expected_daily_returns_[1], k=num_returns_1)
                        if p_return_1 < MIN_PROB:
                            continue
                        prob = p_rental_0 * p_return_0 * p_rental_1 * p_return_1
                        if prob < MIN_PROB:
                            continue
                        # print(prob, p_rental_0, p_return_0, p_rental_1, p_return_1)
                        reward = (num_rentals_0 + num_rentals_1) * self.rental_revenue_per_car_ - abs(action) * self.moving_cost_per_car_
                        expected_reward += prob * (reward + gamma * v[state])
                        # print(reward, expected_reward)
        return expected_reward
                    

    def populate_precomputed_transitions_per_location_(self):
        assert self.precomputed_transitions_per_location_ is None
        self.precomputed_transitions_per_location_ = [[], []]
        # Compute for the two locations
        for i in [0, 1]:
            for num_rentals in range(min(self.max_cars_, UPPER_BOUND_POISSON) + 1):
                p_rental = self.poisson_pmf_(
                    mu=self.expected_daily_rentals_[i], k=num_rentals)
                for num_returns in range(min(self.max_cars_, UPPER_BOUND_POISSON) + 1):
                    p_return = self.poisson_pmf_(
                        mu=self.expected_daily_returns_[i], k=num_returns)
                    self.precomputed_transitions_per_location_[i].append((
                        num_rentals, num_returns, p_rental * p_return))


    def adjust_state_(self, count, rentals, returns):
        assert rentals >= 0 and rentals <= count
        assert returns >= 0
        return min(max(count - rentals + returns, 0), self.max_cars_)


    def poisson_pmf_(self, mu, k):
        if mu not in self.poisson_cache_:
            self.poisson_cache_[mu] = {}
        if k not in self.poisson_cache_[mu]:
            self.poisson_cache_[mu][k] = sp.stats.poisson.pmf(mu=mu, k=k)
        return self.poisson_cache_[mu][k]
    

    def move_cars_(self, state, action):
        new_state = [state[0], state[1]]
        actually_moved = action
        if action > 0:
            actually_moved = min(action, state[0])
        elif action < 0:
            actually_moved = -min(abs(action), state[1])

        new_state[0] -= actually_moved
        new_state[1] += actually_moved
        new_state[0] = min(max(new_state[0], 0), self.max_cars_)
        new_state[1] = min(max(new_state[1], 0), self.max_cars_)
        return (new_state[0], new_state[1]), actually_moved
