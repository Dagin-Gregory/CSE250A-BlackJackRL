import time
import random
import helpers

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

        dealer_value = helpers.evaluate_hand(self.dealer_cards)
        player_value = helpers.evaluate_hand(self.player_cards)
        # Check for natural black jack, immediate 2 card 21
        if (player_value == 21 and len(self.player_cards) == 2):
            self.ongoing = False
            return 1

        if (action == 0):
            # Game is over, someone will win or end in a tie
            self.ongoing = False
            
            # Dealer draws until atleast 17 then they stand
            while (dealer_value < 17):
                self.dealer_cards.append(self.draw_card())
                dealer_value = helpers.evaluate_hand(self.dealer_cards)

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
            player_value = helpers.evaluate_hand(self.player_cards)

            # Player busted
            if (player_value > 21):
                self.ongoing = False
                return -1
            
            if (player_value <= 21):
                return 2