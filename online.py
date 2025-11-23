import time
import random

class game:
    dealer_cards = []
    player_cards = []
    ongoing = 0
    game_rand = random.Random(0)
    def __init__(self):
        self.dealer_cards = []
        self.player_cards = []
        self.ongoing = 0
        self.game_rand = random.Random(time.time_ns())
    
    def draw_card(self):
        # Chooses from 1 of 13 cards
        # Face cards and 10 all count as the same value and don't have any special rules so they get combined into a single value
        card = self.game_rand.randint(1, 13)
        if (card > 10):
            card = 10
        return card
    
    def evaluate_hand(self, hand):
        aces = 0
        value = 0
        for card in hand:
            if (card != 1):
                value += card
            else:
                aces += 1

        while value <= 21 and aces > 0:
            if (value + 11 <= 21):
                value += 11
            else:
                value += 1
            aces -= 1

        return value

    def get_player_cards(self):
        return self.player_cards

    def get_dealer_cards(self):
        return self.dealer_cards

    def new_game(self):
        if (self.ongoing == 0):
            self.ongoing = 1
            self.dealer_cards = []
            self.player_cards = []
            self.game_rand = random.Random(time.time_ns())
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
        if (ongoing == 0):
            raise Exception("Didn't start new game before calling act")
        
        if (action != 0 or action != 1):
            raise Exception("Invalid action")
        # Game can't start and be over so no need to check

        dealer_value = self.evaluate_hand(self.dealer_cards)
        player_value = self.evaluate_hand(self.player_cards)
        # Check for natural black jack, immediate 2 card 21
        if (player_value == 21 and len(self.player_cards) == 2):
            ongoing = 0
            return 1

        if (action == 0):
            # Game is over, someone will win or end in a tie
            ongoing = 0
            # Dealer wins as player stood when dealer had higher value
            if (dealer_value > player_value):
                return -1
            
            # Dealer draws until atleast 17 then they stand
            while (dealer_value <= 17):
                self.dealer_cards.append(self.draw_card())
                dealer_value = self.evaluate_hand(self.dealer_cards)

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
            player_value = self.evaluate_hand(self.player_cards)

            # Player busted
            if (player_value > 21):
                return -1
            
            if (player_value <= 21):
                return 2

        
class RL_Agent_Naive:
    states = {}
    default_probabilities = [1, 1]
    rl_rand = random.Random(0)
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
        player_hand,dealer_hand: unsorted array, [1-10]

        output:
        single value that is a key for the 'states' dictionary
        """
        sorted_player = sorted(player_hand)
        sorted_dealer = sorted(dealer_hand)

        hash_tuple = (hash(str(sorted_player)),
                      hash(str(sorted_dealer)))

        states_hash = hash(hash_tuple)

        return states_hash
    
    def action_probabilities(self, player_hand, dealer_hand):
        """
        return:
            [p_stand, p_hit, ...]
        """
        state_hash = self.hands_to_hash(player_hand, dealer_hand)
        table = self.states.get(state_hash, self.default_probabilities)
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
            if ((prev_probs < rand_action_num or rand_action_num == 0.0) and
                prev_probs + action_probability <= rand_action_num):
                return i
            prev_probs += action_probability
        
        return len(action_probabilities - 1)

    def update_agent(self, states_visited, outcome):
        """
        Adds one to the corresponding actions taken at each state if the outcome of the game is good
        Otherwise adds one to the other action at every state
        input:
            states_visited+action: ( (hands, a from {0, 1, ...})_1, ...)
            states_visited+action: (( (player_hand, dealer_hand), a from {0, 1, ...} )_1, ...)
            outcome: -1, 1 (lost, won)
        returns:
            N/A
        """

        for state in states_visited:
            hands, taken_action = state
            p_hand, d_hand = hands
            hand_hash = self.hands_to_hash(p_hand, d_hand)
            if (hand_hash not in self.states):
                self.states[hand_hash] = self.default_probabilities

            state = self.states[hand_hash]

            if (outcome == 1):
                state[taken_action] += 1
            else:
                dist_value = len(state)-1
                for action in range(len(state)):
                    if(action != taken_action):
                        state[action] += dist_value



def main():
    #num_iterations = 100
    num_iterations = 1
    bj_game = game()
    naive_agent = RL_Agent_Naive()
    while(num_iterations > 0):
        num_iterations -= 1
        bj_game.new_game()
        result = 1
        states = []
        while(result == 1):
            state = (bj_game.get_player_cards(), bj_game.get_dealer_cards())
            agent_action = naive_agent.get_action_learning(state)
            states.append((state, agent_action))
            result = bj_game.act(agent_action)
            


if __name__ == '__main__':
    main()