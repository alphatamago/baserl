import sys

from baserl.common import *
from baserl.graphs import *
from baserl.mdp_base import MDPBase

class JacksRental(MDPBase):
    """
    Jack has two rental locations, with max_cars that can be parked overnight
    in each. Each day, some people come to rent cars (with a Poisson probability
    expected_daily_rentals) and some return cars previously rented (with Poisson
    probability expected_daily_returns).
    Jack receives rental_revenue_per_car for each rental.
    Jack can move cars between the two locations overnight at a cost of
    moving_cost_per_car per each moved car. There is a limit to the number of
    such overnight moves of max_moves.

    The goal is to learn an optimal policy, that maximizes future rewards,
    discounted by 0.9 in the original problem formulation.
    """

    def __init__(self, 
                 max_cars=10, 
                 max_moves=5,
                 rental_revenue_per_car=10,
                 moving_cost_per_car=2,
                 expected_daily_rentals=[3, 4],
                 expected_daily_returns = [3, 2],
                 transitions_prob_threshold = 10):
        """
        Most of the parameters are described in the class level comment above.

        transitions_prob_threshold controls how much of the actual probabilty
        distribution about (number_rental, number_returns) that can occur at
        each of the rental locations daily, we want to keep.
        If we keep it all, with max_cars=10 it will take several minutes to
        run policy iteration, and maybe hours for max_cars=20.
        We can specify either an interger - in which case that is the number of
        the top pairs to keep - or a float between 0.0 and 1.0, in which case
        that is the ratio of the probabily mass to keep.
        In either case, the probability mass of the discarded iteams is being
        residtributed amoung the ones we keep, proportional to the probabilities
        in the raw distribution.
        """

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

        self.transitions_prob_threshold_ = transitions_prob_threshold
                 
        # Caching the transitions information, to avoid expensive, repeated
        # computations
        self.global_transitions_ = {}
        self.global_transitions_hits_ = 0
        self.global_transitions_misses_ = 0

        self.precomputed_transitions_per_location_ = None
                 

    def states(self):
        """
        Returns all the possible states.
        """
        return   [(x, y) for x in range(0, self.max_cars_ + 1)
                  for y in range(0, self.max_cars_ + 1)]


    def actions(self, state):
        """
        What actions can one take from a given state.
        """
        return range(-min(state[1], self.max_moves_), min(state[0],
                                                          self.max_moves_) + 1)


    def is_terminal(self, state):
        # There is no terminal state in this problem
        return False
    

    def gamma(self):
        return 0.9
    

    def transitions(self, evening_state, action):
        """
        (evening_state, action) -> (state) -> [(new_state, reward, prob)]
        evening_state = (num_0, num_1) = counts at the two locations at the end
        of day
        action = car moved from 0 to 1 (or 1 to 0, if negative)
        reward = revenue from rentals the next day, minus fees for incurred by
        moving cars (etc)
        new_state = counts at the end of next day, taking into account the
        overnight moves, rentals and returns next day
        """

        if self.precomputed_transitions_per_location_ is None:
            self.populate_precomputed_transitions_per_location_() 
        
        assert type(evening_state) == tuple and len(evening_state) == 2
        assert (evening_state[0] >= 0 and evening_state[0] <= self.max_cars_ and
                evening_state[1] >= 0 and evening_state[1] <= self.max_cars_)
        assert action >= -self.max_moves_ and self.max_moves_ <= self.max_moves_

        if action > 0:
            assert action <= evening_state[0]
        elif action <0:
            assert abs(action) <= evening_state[1]
        
        assert len(self.precomputed_transitions_per_location_[0]) > 0
        assert len(self.precomputed_transitions_per_location_[1]) > 0

        # We get the updated state after moving cars, and the actual number of
        # moves
        state, actually_moved = self.move_cars_(evening_state, action)
        if action >= 0:
            assert actually_moved >=0 and actually_moved <= action
        else:
            assert actually_moved <= 0 and actually_moved >= action

        # cache_key = (evening_state, action)
        cache_key = state
        
        if cache_key in self.global_transitions_:
            self.global_transitions_hits_ += 1
        else:
            self.global_transitions_misses_ += 1
            self.global_transitions_[cache_key] = {}
            sum_probs = 0
            for (num_rentals_0, num_returns_0, prob_0) in self.precomputed_transitions_per_location_[0]:
                assert prob_0 > 0
                num_rentals_0 = min(num_rentals_0, state[0])
                for (num_rentals_1, num_returns_1, prob_1) in self.precomputed_transitions_per_location_[1]:
                    assert prob_1 > 0
                    sum_probs += prob_0 * prob_1
                    num_rentals_1 = min(num_rentals_1, state[1])
                    assert (num_rentals_0 >= 0 and num_rentals_0 <=
                            self.max_cars_ and num_rentals_1 >= 0 and
                            num_rentals_1 <= self.max_cars_)
                    assert (num_returns_0 >= 0 and num_returns_0 <=
                            self.max_cars_
                            and num_returns_1 >= 0 and num_returns_1 <=
                            self.max_cars_)
                    reward = ((num_rentals_0 + num_rentals_1) *
                              self.rental_revenue_per_car_)
                    outcome = (((self.adjust_state_(state[0], num_rentals_0,
                                                    num_returns_0),
                             self.adjust_state_(state[1], num_rentals_1,
                                                num_returns_1)), reward),
                               prob_0 * prob_1)
                    if outcome[0] not in self.global_transitions_[cache_key]:
                        self.global_transitions_[cache_key][outcome[0]] = 0
                    self.global_transitions_[cache_key][outcome[0]] += outcome[1]

        # Regardless of whether freshly computed or fetched from cache, we need
        # to adjust the rewards based on the cost of moving cars
        result = list(self.global_transitions_[cache_key].items())

        # DEBUG only code
        sum_vals = sum([v for _, v in result])
        if abs(sum_vals-1.0) >= 0.01:
            print ('expected sum probs 1, got:', sum_vals, 'state:', state,
                   'action:', action)
            assert(False)

        return [((k[0][0], k[0][1] - abs(actually_moved) *
                  self.moving_cost_per_car_), k[1]) for k in result]


    def print_value(self, v):
        heatmap_value_function(v)
        #for x in range(self.max_cars_):
        #    for y in range(self.max_cars_):
        #        sys.stdout.write("%3.2f " % v[(x, y)])
        #    sys.stdout.write("\n")


    def print_policy(self, policy):
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


    def report_stats(self):
        for k in self.poisson_cache_.keys():
            print("poisson_cache key:", k, "len:", len(self.poisson_cache_[k]))
        print("max_cars:", self.max_cars_)
        print("max_moves:", self.max_moves_)
        print("rental_revenue_per_car:", self.rental_revenue_per_car_)
        print("moving_cost_per_car", self.moving_cost_per_car_)
        print("expected_daily_rentals:", self.expected_daily_rentals_)
        print("expected_daily_returns:", self.expected_daily_returns_)
        print("len global_transitions:", len(self.global_transitions_))
        for k in list(self.global_transitions_.keys())[:3]:
            print("global_transitions key:", k, "len:",
                  len(self.global_transitions_[k]))
        if self.precomputed_transitions_per_location_ is not None:
            for d in self.precomputed_transitions_per_location_:
                print("precomputed_transitions_per_location_:", len(d))
        print("global_transitions_hits:", self.global_transitions_hits_)
        print("global_transitions_misses:", self.global_transitions_misses_)
        print("global_transitions hit ratio:",
              self.global_transitions_hits_/(self.global_transitions_misses_ +
                                             self.global_transitions_hits_ + 1))


    def populate_precomputed_transitions_per_location_(self):
        assert self.precomputed_transitions_per_location_ is None
        self.precomputed_transitions_per_location_ = [[], []]
        # Compute for the two locations
        for i in [0, 1]:
            for num_rentals in range(self.max_cars_ + 1):
                p_rental = self.poisson_pmf_(mu=self.expected_daily_rentals_[i],
                                             k=num_rentals)
                for num_returns in range(self.max_cars_ + 1):
                    p_return = self.poisson_pmf_(
                        mu=self.expected_daily_returns_[i], k=num_returns)
                    self.precomputed_transitions_per_location_[i].append(
                        (num_rentals, num_returns, p_rental * p_return))
        # Remove the items with very low probabilities, re-distribute their
        # prob mass to the survivors
        for i in [0, 1]:
            len_before = len(self.precomputed_transitions_per_location_[i])
            self.precomputed_transitions_per_location_[i] = chop_prob_distribution(self.precomputed_transitions_per_location_[i], self.transitions_prob_threshold_, 2)
            len_after = len(self.precomputed_transitions_per_location_[i])
            #print("location", i, "len before prob chop:", len_before, "after:",
            #      len_after, "sample top:",
            #      self.precomputed_transitions_per_location_[i][:5],
            #      "sample bottom:",
            #      self.precomputed_transitions_per_location_[i][-5:])


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
