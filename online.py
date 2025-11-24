import time
import random

num_games = 1000000

def evaluate_hand(hand):
    """
    Returns the maximum value of the hand, taking the multiple values an ace can take into account

    input: array of cards stored as integers, [1-13]
    return: integer value of all cards in the hand
    """
    aces = 0
    value = 0
    for card in hand:
        if (card != 1 and card < 10):
            value += card
            continue
        if (card >= 10):
            value += 10
            continue
        if (card == 1):
            aces += 1
            continue

    while aces > 0:
        if (value + 11 <= 21):
            value += 11
        else:
            value += 1
        aces -= 1

    return value

class game:
    dealer_cards = []
    player_cards = []
    previous_seed = time.time_ns()
    ongoing = False
    game_rand = random.Random(previous_seed)
    def __init__(self):
        self.dealer_cards = []
        self.player_cards = []
        self.ongoing = False
        self.previous_seed = time.time_ns()
        self.game_rand = random.Random(self.previous_seed)
    
    def draw_card(self):
        # Chooses from 1 of 13 cards
        # Face cards and 10 all count as the same value and don't have any special rules so they get combined into a single value
        card = self.game_rand.randint(1, 13)
        return card

    def get_player_cards(self):
        return self.player_cards.copy()

    def get_dealer_cards(self):
        return self.dealer_cards.copy()

    def new_game(self):
        if (self.ongoing == False):
            self.ongoing = True
            self.dealer_cards = []
            self.player_cards = []

            # Dealer only starts with one card from the deck since drawing cards are independent of eachother, starting with 1 card face down
            # Is the same as not drawing a card
            self.dealer_cards.append(self.draw_card())

            # Player always starts with 2
            self.player_cards.append(self.draw_card())
            self.player_cards.append(self.draw_card())

    def act(self, action):
        """
        inputs
        action: 0=stand, 1=hit

        returns
        -1=dealer won, 0=tie, 1=agent won 2=ongoing
        """
        if (self.ongoing == False):
            raise Exception("Didn't start new game before calling act")
        
        if (action != 0 and action != 1):
            raise Exception("Invalid action: ", action)
        # Game can't start and be over so no need to check

        dealer_value = evaluate_hand(self.dealer_cards)
        player_value = evaluate_hand(self.player_cards)
        # Check for natural black jack, immediate 2 card 21
        if (player_value == 21 and len(self.player_cards) == 2):
            self.ongoing = False
            return 1

        if (action == 0):
            # Game is over, someone will win or end in a tie
            self.ongoing = False
            # Dealer wins as player stood when dealer had higher value
            if (dealer_value > player_value):
                return -1
            
            # Dealer draws until atleast 17 then they stand
            while (dealer_value < 17):
                self.dealer_cards.append(self.draw_card())
                dealer_value = evaluate_hand(self.dealer_cards)

            # Dealer busted, agent wins
            if (dealer_value > 21):
                return 1
            
            # Dealer got blackjack
            if (dealer_value == 21 or dealer_value > player_value):
                return -1
            
            if (dealer_value < player_value):
                return 1
            
            if (dealer_value == player_value):
                return 0
            
        if (action == 1):
            self.player_cards.append(self.draw_card())
            player_value = evaluate_hand(self.player_cards)

            # Player busted
            if (player_value > 21):
                self.ongoing = False
                return -1
            
            if (player_value <= 21):
                return 2

default_probabilities = [1, 1]

class RL_Agent:
    states = {}
    rl_rand = random.Random(time.time_ns())
    """
    Every state will be a mapping, from a (player_cards, dealer_cards) => (P(Hit), P(Stand))
    In order to capture position invariance of the set of cards, the states encode the cards in 
    """
    def __init__(self):
        self.states = {}
        self.rl_rand = random.Random(time.time_ns())

    def hands_to_hash(self, player_hand, dealer_hand):
        """
        inputs:
        player_hand,dealer_hand: unsorted array, [1-13]

        output:
        single value that is a key for the 'states' dictionary
        """
        player_value = evaluate_hand(player_hand)
        dealer_value = evaluate_hand(dealer_hand)

        hash_tuple = (hash(player_value),
                      hash(dealer_value))

        states_hash = hash(hash_tuple)

        return states_hash
    
    def action_probabilities(self, player_hand, dealer_hand):
        """
        return:
            [p_stand, p_hit, ...]
        """
        state_hash = self.hands_to_hash(player_hand, dealer_hand)
        table = self.states.get(state_hash, default_probabilities)
        total = 0.0
        for count in table:
            total += count

        probs = []
        for count in table:
            probs.append(count / total)

        return probs

    def get_action_learning(self, state):
        """
        Will choose between a random action 
        inputs:
            state: (player_hand, dealer_hand)
        return:
            action: (0, 1, ...)
        """
        p_hand, d_hand = state
        action_probabilities = self.action_probabilities(p_hand, d_hand)
        prev_probs = 0.0
        rand_action_num = self.rl_rand.random()
        for i in range(len(action_probabilities)):
            action_probability = action_probabilities[i]
            if (rand_action_num <= prev_probs + action_probability):
                return i
            prev_probs += action_probability
        
        return len(action_probabilities) - 1

    def update_agent(self, states_visited, outcome):
        """
        Adds one to the corresponding actions taken at each state if the outcome of the game is good
        Otherwise adds one to the other action at every state
        input:
            states_visited+action: ( (hands, a from {0, 1, ...})_1, ...)
            states_visited+action: (( (player_hand, dealer_hand), a from {0, 1, ...} )_1, ...)
            outcome: -1, 0, 1 (lost, tie, won)
        returns:
            N/A
        """

        for visited_state in states_visited:
            hands, taken_action = visited_state
            p_hand, d_hand = hands
            hand_hash = self.hands_to_hash(p_hand, d_hand)
            if (hand_hash not in self.states):
                copy_arr = []
                for val in default_probabilities:
                    copy_arr.append(val)
                self.states[hand_hash] = copy_arr

            state = self.states[hand_hash]

            if (outcome == 1):
                state[taken_action] += 1
                continue
            if (outcome == -1):
                dist_value = len(state)-1
                for action in range(len(state)):
                    if(action != taken_action):
                        state[action] += dist_value
                continue
            # Tie game, don't update policy as no useful information can be gained
            if (outcome == 0):
                continue

class RL_Agent_Naive(RL_Agent):
    # Winrate:  35.91 %  |  Tierate:  5.02 %  |  Lossrate:  59.07 % | 1000000 games

    def hands_to_hash(self, player_hand, dealer_hand):
        """
        inputs:
        player_hand,dealer_hand: unsorted array, [1-13]

        output:
        single value that is a key for the 'states' dictionary
        """
        sorted_player = sorted(player_hand)
        sorted_dealer = sorted(dealer_hand)

        hash_tuple = (hash(str(sorted_player)),
                      hash(str(sorted_dealer)))

        states_hash = hash(hash_tuple)

        return states_hash


class RL_Agent_Naive_ValueBased(RL_Agent):
    # 36.26 %  |  Tierate:  5.06 %  |  Lossrate:  58.67 % | 1000000 games
    
    def hands_to_hash(self, player_hand, dealer_hand):
        """
        inputs:
        player_hand,dealer_hand: unsorted array, [1-13]

        output:
        single value that is a key for the 'states' dictionary
        """
        player_value = evaluate_hand(player_hand)
        dealer_value = evaluate_hand(dealer_hand)

        hash_tuple = (hash(player_value),
                      hash(dealer_value))

        states_hash = hash(hash_tuple)

        return states_hash


def main():
    print_hands = False
    num_iterations = num_games
    #num_iterations = 100
    games_played = num_iterations

    bj_game = game()
    #agent = RL_Agent_Naive()
    agent = RL_Agent_Naive_ValueBased()
    wins = 0.0
    losses = 0.0
    ties = 0.0

    while(num_iterations > 0):
        num_iterations -= 1
        if (num_iterations == 0):
            print("Debug statement")
        bj_game.new_game()
        
        result = 2
        states = []
        while(result == 2):
            state = (bj_game.get_player_cards(), bj_game.get_dealer_cards())
            agent_action = agent.get_action_learning(state)
            states.append((state, agent_action))
            result = bj_game.act(agent_action)
        agent.update_agent(states, result)
        agent_hand = bj_game.get_player_cards()
        dealer_hand = bj_game.get_dealer_cards()
        if (result == 1):
            wins += 1
            if (print_hands):
                print("Won, player hand: ", agent_hand, " | dealer hand: ", dealer_hand)
            continue
        if (result == 0):
            ties += 1
            if (print_hands):
                print("Tied, player hand: ", agent_hand, " | dealer hand: ", dealer_hand)
            continue
        if (result == -1):
            losses += 1
            if (print_hands):
                print("Lost, player hand: ", agent_hand, " | dealer hand: ", dealer_hand)
            continue
    
    print("Training completed for ", games_played, " iterations.")
    print("Winrate: ",  f"{(wins / games_played) * 100:.2f}", "%", " | ",
          "Tierate: ",  f"{(ties / games_played) * 100:.2f}", "%", " | ",
          "Lossrate: ", f"{(losses / games_played) * 100:.2f}", "%")


if __name__ == '__main__':
    main()