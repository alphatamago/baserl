import sys

from baserl.common import *

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
    

    def transitions(self, state, action):
        """
        (state, action) -> [(new_state, reward, prob)]
        state = (num_0, num_1) = counts at the two locations at the end of day
        action = car moved from 0 to 1 (or 1 to 0, if negative)
        reward = revenue from rentals the next day, minus fees for incurred by moving cars (etc)
        new_state = counts at the end of next day, taking into account the overnight moves, rentals and returns next day
        """

        if self.precomputed_transitions_per_location_ is None:
            self.populate_precomputed_transitions_per_location_() 
        
        assert type(state) == tuple and len(state) == 2
        assert state[0] >= 0 and state[0] <= self.max_cars_ and state[1] >= 0 and state[1] <= self.max_cars_
        assert action >= -self.max_moves_ and self.max_moves_ <= self.max_moves_

        if (state, action) in self.global_transitions_:
            return list(self.global_transitions_[(state, action)].items())

        assert len(self.precomputed_transitions_per_location_[0]) > 0
        assert len(self.precomputed_transitions_per_location_[1]) > 0
        # print(len(transitions_per_location[0]), len(transitions_per_location[1]))

        (state_0, state_1), _ = self.move_cars_(state, action)

        self.global_transitions_[(state, action)] = {}
        sum_probs = 0
        for (num_rentals_0, num_returns_0, prob_0) in self.precomputed_transitions_per_location_[0]:
            assert prob_0 > 0
            num_rentals_0 = min(num_rentals_0, state_0)
            for (num_rentals_1, num_returns_1, prob_1) in self.precomputed_transitions_per_location_[1]:
                assert prob_1 > 0
                sum_probs += prob_0 * prob_1
                num_rentals_1 = min(num_rentals_1, state_1)
                assert (num_rentals_0 >= 0 and num_rentals_0 <= self.max_cars_ and num_rentals_1 >= 0 and 
                        num_rentals_1 <= self.max_cars_)
                assert (num_returns_0 >= 0 and num_returns_0 <= self.max_cars_ and num_returns_1 >= 0 and 
                        num_returns_1 <= self.max_cars_)
                reward = (num_rentals_0 + num_rentals_1) * self.rental_revenue_per_car_ - abs(action) * self.moving_cost_per_car_

                outcome = (((self.adjust_state_(state_0, num_rentals_0, num_returns_0),
                         self.adjust_state_(state_1, num_rentals_1, num_returns_1)), reward), prob_0*prob_1)
                if outcome[0] not in self.global_transitions_[(state, action)]:
                    self.global_transitions_[(state, action)][outcome[0]] = 0
                self.global_transitions_[(state, action)][outcome[0]] += outcome[1]

        result = list(self.global_transitions_[(state, action)].items())
        sum_vals = sum([v for _, v in result])
        """
        if sum_probs != sum_vals:
            print (sum_probs, sum_vals)
            assert (False)
        """
        if abs(sum_vals-1.0) >= 0.01:
            print ('expected sum probs 1, got:', sum_vals, 'state:', state, 'action:', action)
            assert(False)
        return result


    def print_value(self, v, states):
        for x in range(self.max_cars_):
            for y in range(self.max_cars_):
                sys.stdout.write("%3.2f " % v[(x, y)])
            sys.stdout.write("\n")
            
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


    def populate_precomputed_transitions_per_location_(self):
        assert self.precomputed_transitions_per_location_ is None
        self.precomputed_transitions_per_location_ = [[], []]
        # Compute for the two locations
        for i in [0, 1]:
            for num_rentals in range(self.max_cars_ + 1):
                p_rental = self.poisson_pmf_(mu=self.expected_daily_rentals_[i], k=num_rentals)
                for num_returns in range(self.max_cars_ + 1):
                    p_return = self.poisson_pmf_(mu=self.expected_daily_returns_[i], k=num_returns)
                    self.precomputed_transitions_per_location_[i].append((num_rentals, num_returns, p_rental * p_return))
                    
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
