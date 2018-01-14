import sys

from baserl.common import *
from baserl.mdp_base import MDPBase
from baserl.monte_carlo import *

BJ_HIT = 0
BJ_STICK = 1


def draw_blackjack_card():
    card = random.randint(1, 13)
    return card if card <= 10 else 10


class BlackjackEpisodeGenerator:
    def __init__(self, with_exploring_starts, verbose=False):
        self.with_exploring_starts_ = with_exploring_starts
        self.verbose_ = verbose

    def generate(self, policy, start_state=None, start_action=None):
        episodes = []

        # Player's turn
        if start_state is None:
            start_state = (random.randint(11, 21), random.randint(1, 10),
                           random.random() < 0.5)
            if self.verbose_:
                print("Generated random start_state", start_state)
        else:
            if self.verbose_:
                print("Fixed start_state", start_state)
            
        current_state = copy.deepcopy(start_state)
            
        is_first = True
        while True:
            new_action = None
            if is_first:
                if start_action is not None:
                    new_action = start_action
                    if self.verbose_:
                        print("Fixed start_action:", new_action)
                elif self.with_exploring_starts_:
                    new_action = random.randint(0, 1)
                    if self.verbose_:
                        print("Random start_action:", new_action)
            is_first = False
            if new_action is None:
                new_action = select_random_legal_action(policy, current_state)
                if self.verbose_:
                        print("Policy action:", new_action)

            episodes.append([current_state, new_action, 0])
            if self.verbose_:
                print("episode:", episodes)
                
            if new_action == BJ_HIT:
                new_card_value = draw_blackjack_card()
                new_player_sum = current_state[0] + new_card_value
                if self.verbose_:
                    print("new card:", new_card_value)
                # TODO - my addition
                if new_card_value == 1 and new_player_sum + 10 <= 21:
                    new_player_sum += 10
                    current_state = (new_player_sum, current_state[1], True)
                    if self.verbose_:
                        print("Used ace, current_state:", current_state)
                else:
                    if new_player_sum > 21:
                        if current_state[2]:
                            new_player_sum -= 10
                            current_state = (new_player_sum, current_state[1],
                                             False)
                            if self.verbose_:
                                print("Saved by the ace, current_state:",
                                      current_state)
                        else:
                            # Mark loss
                            episodes[-1][2] = -1
                            if self.verbose_:
                                print("Player busted, episode:", episodes)
                            return episodes
                    else:
                        # TODO - don't we need to add this to episodes???
                        current_state = (new_player_sum, current_state[1],
                                         current_state[2])
                        if self.verbose_:
                                print("current_state:", current_state)
            else:
                assert new_action == BJ_STICK
                break

        # Dealer's turn
        if start_state[1] == 1:
            dealer_sum = 11
            dealer_has_usable_ace = True
            if self.verbose_:
                print("dealer has usable_ace, dealer_sum=", dealer_sum)
        else:
            dealer_sum = start_state[1]
            dealer_has_usable_ace = False
            if self.verbose_:
                print("dealer doesn't have usable_ace, dealer_sum=", dealer_sum)

        while dealer_sum < 17:
            new_card_value = draw_blackjack_card()
            dealer_sum += new_card_value
            if self.verbose_:
                print("dealer new card", new_card_value, "dealer_sum=",
                      dealer_sum)
            if dealer_sum > 21 and dealer_has_usable_ace:
                dealer_sum -= 10
                dealer_has_usable_ace = False
                if self.verbose_:
                    print("dealer used ace, dealer_sum=", dealer_sum)

        if dealer_sum > 21:
            episodes[-1][2] = 1
            if self.verbose_:
                    print("dealer busted!")
        else:
            dealer_gap = abs(dealer_sum - 21)
            player_gap = abs(current_state[0] - 21)
            if self.verbose_:
                    print("game over, player_gap=", player_gap, "dealer_gap=",
                          dealer_gap)
            if player_gap < dealer_gap:
                episodes[-1][2] = 1
                if self.verbose_:
                    print("player wins!")
            elif player_gap > dealer_gap:
                episodes[-1][2] = -1
                if self.verbose_:
                    print("dealer wins!")
            else: # player_gap == dealer_gap           
                episodes[-1][2] = 0
                if self.verbose_:
                    print("nobody wins!")

        return episodes

class Blackjack(MDPBase):
    """
    The game of Blackjack, as described in Sutton's RL book 2nd edition, page 76.

    This is modeled as an undiscounted episodic MDP.
    """

    def __init__(self):
        pass


    def gamma(self):
        return 1.0


    def states(self):
        # value 22 stands for any bust state
        return [(ps, d, ua) for ps in range(2, 23) for d in range(1, 11)
                for ua in [0, 1]] 


    def actions(self, state):
        return  [BJ_HIT, BJ_STICK]


    def make_simple_blackjack_player_policy(self):
        """
        Hit until the sum is either 20 or 21, then stick.
        """
        policy = {}
        for (sum_player, visible_card_dealer, usable_ace) in self.states():
            if sum_player < 20:
                policy[(sum_player, visible_card_dealer, usable_ace)] = {
                    BJ_HIT:1.0,
                    BJ_STICK:0.0}
            else:
                policy[(sum_player, visible_card_dealer, usable_ace)] = {
                    BJ_HIT:0.0,
                    BJ_STICK:1.0}
        return policy
