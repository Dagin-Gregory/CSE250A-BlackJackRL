import time
import random
import helpers

class game:
    init_BUST_THRESHOLD = 21
    init_DEALER_HIT_THRESHOLD = 17

    dealer_cards = []
    player_cards = []
    previous_seed = time.time_ns()
    ongoing = False
    game_rand = random.Random(previous_seed)
    def __init__(self,
                 bust_thresh=21,
                 hit_thresh=17):
        self.dealer_cards = []
        self.player_cards = []
        self.ongoing = False
        self.previous_seed = time.time_ns()
        self.game_rand = random.Random(self.previous_seed)
        self.init_BUST_THRESHOLD = bust_thresh
        self.init_DEALER_HIT_THRESHOLD = hit_thresh
    
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
            return False

    def act(self, action):
        """
        inputs
        action: 0=stand, 1=hit

        returns
        -1=dealer won, 0=tie, 1=agent won 2=ongoing
        """
        if (self.ongoing == False):
            raise Exception("Didn't start new game before calling act")
        
        if (not(action is helpers.actions.STAND or
                action is helpers.actions.HIT or
                action is helpers.actions.DOUBLE_DOWN or
                action is helpers.actions.SPLIT)):
            raise Exception("Invalid action: ", action)
        # Game can't start and be over so no need to check

        dealer_value = helpers.evaluate_hand(self.dealer_cards)
        player_value = helpers.evaluate_hand(self.player_cards)
        # Check for natural black jack, immediate 2 card 21
        if (player_value == self.init_BUST_THRESHOLD and len(self.player_cards) == 2):
            self.ongoing = False
            return helpers.result.AGENT_WON

        if (action == helpers.actions.STAND):
            # Game is over, someone will win or end in a tie
            self.ongoing = False
            
            # Dealer draws until atleast 17 then they stand
            while (dealer_value < self.init_DEALER_HIT_THRESHOLD):
                self.dealer_cards.append(self.draw_card())
                dealer_value = helpers.evaluate_hand(self.dealer_cards)

            # Dealer busted, agent wins
            if (dealer_value > self.init_BUST_THRESHOLD):
                return helpers.result.AGENT_WON
            
            # Dealer got blackjack
            if (dealer_value == self.init_BUST_THRESHOLD or dealer_value > player_value):
                return helpers.result.DEALER_WON
            
            if (dealer_value < player_value):
                return helpers.result.AGENT_WON
            
            if (dealer_value == player_value):
                return helpers.result.TIE
            
        if (action == helpers.actions.HIT):
            self.player_cards.append(self.draw_card())
            player_value = helpers.evaluate_hand(self.player_cards)

            # Player busted
            if (player_value > self.init_BUST_THRESHOLD):
                self.ongoing = False
                return helpers.result.DEALER_WON
            
            if (player_value <= self.init_BUST_THRESHOLD):
                return helpers.result.ONGOING


class finite_deck_game(game):
    total_decks = 3
    reshuffle_threshold = .15
    dealer_hole_card = 0
    card_list = []

    def __init__(self,
                 bust_thresh=21,
                 hit_thresh=17,
                 num_decks=3,
                 reshuffle_thresh=.15):
        super().__init__(bust_thresh, hit_thresh)
        self.total_decks = num_decks
        self.reshuffle_threshold = reshuffle_thresh
        self.card_list = []
        self.dealer_hole_card = 0

    def shuffle_deck(self):
        self.card_list = []
        cards_left = []
        for i in range(1, 13+1):
            for j in range(self.total_decks * 4):
                cards_left.append(i)
        
        while (len(cards_left) > 0):
            rand_index = self.game_rand.randint(0, len(cards_left)-1)
            self.card_list.append(cards_left[rand_index])
            _ = cards_left.pop(rand_index)

    def draw_card(self):
        top_card = self.card_list.pop()
        return top_card
    
    def reshuffle_if_necessary(self):
        if (len(self.card_list) <= self.reshuffle_threshold * self.total_decks * 52):
            self.shuffle_deck()
            return True
        return False
    
    def new_game(self):
        reshuffled = self.reshuffle_if_necessary()
        self.dealer_hole_card = self.draw_card()
        super().new_game()
        return reshuffled

    def act(self, action):
        if (self.dealer_hole_card != 0):
            self.dealer_cards.append(self.dealer_hole_card)
            self.dealer_hole_card = 0

        value = super().act(action)
        return value