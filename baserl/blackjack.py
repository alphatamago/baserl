import sys

from baserl.common import *
from baserl.mdp_base import MDPBase
from baserl.monte_carlo import *

def normalize_card(card):
    """
    We take 1 as the representation for ace
    """
    assert card >=1
    assert card <= 14
    assert card != 11
    if card >= 12:
        return 10
    else:
        return card

assert(normalize_card(1) == 1)
assert(normalize_card(10) == 10)
assert(normalize_card(12) == 10)
assert(normalize_card(13) == 10)
assert(normalize_card(14) == 10)


def normalize_hand(original_cards):
    cards = []
    for c in original_cards:
        cards.append(normalize_card(c))
    return cards

assert(normalize_hand([1, 2, 10, 14, 13]) == [1, 2, 10, 10, 10])

def check_win(cards):
    # the cards have already been normalized
    for c in cards:
        assert c >=1 and c <=10
    s = sum(cards)
    if s > 21:
        return False
    if s == 21:
        return True
    # Check for ace
    if 1 in cards and (s + 10  == 21):
        return True
    return False

assert (check_win([1, 10]) == True)
assert (check_win([1] * 21) == True)
assert (check_win([1] * 11) == True)
assert (check_win([5] + [1] * 6) == True)
assert (check_win([10, 1]) == True)
assert (check_win([10, 10, 1]) == True)
assert (check_win([10, 10, 2]) == False)
assert (check_win([10, 10]) == False)
assert (check_win([10, 10]) == False)
assert (check_win([]) == False)


def make_state(player_cards, dealer_cards):
    encoded_player_cards = []
    s = sum(player_cards)
    # Check for "usable ace"
    if 1 in player_cards and s + 10 <= 21:
        return (s+10, dealer_cards[0], True)
    else:
        return (min(22, s), dealer_cards[0], False)


blackjack_cards = list(range(1, 11)) + list(range(12, 15))
def draw_blackjack_card():
    return blackjack_cards[random.randint(0, len(blackjack_cards)-1)]


class BlackjackEpisodeGenerator:
    """
    If with_exploring_starts is True, we need to ensure that all (state, actions)
    pairs have a chance to be at the beginning of episodes.
    We draw cards randomly so that should ensure the start randomness; we need to do
    something special for action randomness though, otherwise the initial policy can
    result in not trying certain actions at all in certain states.
    """
    
    def __init__(self, with_exploring_starts, verbose=False):
        self.with_exploring_starts_ = with_exploring_starts
        self.verbose_ = verbose

    def generate(self, policy, start_state=None,
                 start_action=None):
        """
        We ignore max_episode_len argument, since we need an episode to be a
        complete game.
        Also, we ignore start_state and start_action, since this episode generator
        is naturally covering all possible states.
        """
        # Drawing cards for the player and the dealer
        player_cards = normalize_hand([draw_blackjack_card(),
                                       draw_blackjack_card()])

        # dealer_cards[0] is visible to the player
        dealer_cards = normalize_hand([draw_blackjack_card(),
                                       draw_blackjack_card()])

        if self.verbose_:
            print("p:", player_cards, "d:", dealer_cards)

        current_state = make_state(player_cards, dealer_cards)
        assert (current_state[0] >= 2)
        assert (current_state[1] < 23)
        assert (current_state[1] >= 1)
        assert (current_state[1] < 11)

        # Check for natural wins
        if check_win(player_cards):
            if check_win(dealer_cards):
                # draw
                if self.verbose_:
                    print("natural draw")
                return [(current_state, BJ_STICK, 0.0)]
            else:
                # player_wins
                if self.verbose_:
                    print("natural player wins")
                return [(current_state, BJ_STICK, 1.0)]
        elif check_win(dealer_cards):
            # dealer wins
            if self.verbose_:
                print("natural dealer wins")
            return [(current_state, BJ_STICK, -1.0)]
        
        if sum(player_cards) > 21:
            # Player went bust
            if self.verbose_:
                print("natural player bust")
            return [(current_state, BJ_STICK, -1.0)]
        
        episode = []
        
        # In the Exploring-Starts case, we pick just the first action randomly.
        if self.with_exploring_starts_:
            # Select a random action
            if random.randint(0,1) == 0:
                player_action = BJ_HIT
            else:
                player_action = BJ_STICK
        else:
            # player's turn: use the policy argument to pick action, u
            player_action = select_random_legal_action(policy, current_state)
        while player_action == BJ_HIT:
            # hit
            player_cards.append(normalize_card(draw_blackjack_card()))
            if self.verbose_:
                print("player HIT")
                print("p:", player_cards, "d:", dealer_cards)

            ps = sum(player_cards)
            if ps <= 21:
                # the game goes on
                episode.append((current_state, BJ_HIT, 0.0))
                if self.verbose_:
                    print("player sum=", ps)
            else:
                assert ps > 21
                episode.append((current_state, BJ_HIT, -1.0))
                # player lost
                if self.verbose_:
                    print("player lost, sum=", ps)
                return episode
            # update state at the end of loop
            current_state = make_state(player_cards, dealer_cards)
            # select next action
            player_action = select_random_legal_action(policy, current_state)

        if self.verbose_:
            print("player STICKS")
                
        # The player chose to stick - it is dealer's turn next
        current_state = make_state(player_cards, dealer_cards)

        # dealer's turn - by rule, stick to sum >= 17, hit otherwise
        ds = sum(dealer_cards)
        if self.verbose_:
            print("dealer sum=", ds)

        while ds < 17:
            # hit
            dealer_cards.append(normalize_card(draw_blackjack_card()))
            ds = sum(dealer_cards)

            if self.verbose_:
                print("dealer HITS")
                print("p:", player_cards, "d:", dealer_cards)
                print("dealer sum=", ds)

        if ds > 21:
            # dealer bust, player wins
            episode.append((current_state, BJ_STICK, 1.0))
            if self.verbose_:
                print("dealer lost, sum=", ds)
            return episode

        # game ends, the one closest to 21 wins
        
        # Check if dealer can use ace
        if 1 in dealer_cards and ds + 10 <= 21:
            ds += 10
            if self.verbose_:
                print("dealer can use ACE, updated sum=", ds)
            
        result = 0.0
        ps = sum(player_cards)
        # Check if player can use ace
        if 1 in player_cards and ps + 10 <= 21:
            ps += 10
            if self.verbose_:
                print("player can use ACE, updated sum=", ps)
        
        if ps > ds:
            result = 1.0
            if self.verbose_:
                print("player wins ps=", ps, "ds=", ds)
        elif ps < ds:
            result = -1.0
            if self.verbose_:
                print("dealer wins ps=", ps, "ds=", ds)
        else:
            if self.verbose_:
                print("draw ps=", ps, "ds=", ds)
        episode.append((current_state, BJ_STICK, result))
        return episode


BJ_HIT = 0
BJ_STICK = 1

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
         return [(ps, d, ua) for ps in range(2, 22) for d in range(1, 14)
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
