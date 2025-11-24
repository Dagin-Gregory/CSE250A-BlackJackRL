import time
import random
import helpers

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
        player_value = helpers.evaluate_hand(player_hand)
        dealer_value = helpers.evaluate_hand(dealer_hand)

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
        player_value = helpers.evaluate_hand(player_hand)
        dealer_value = helpers.evaluate_hand(dealer_hand)

        hash_tuple = (hash(player_value),
                      hash(dealer_value))

        states_hash = hash(hash_tuple)

        return states_hash