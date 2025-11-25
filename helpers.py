from enum import Enum

class actions(Enum):
    STAND = 0
    HIT = 1
    DOUBLE_DOWN = 2
    SPLIT = 3

class result(Enum):
    DEALER_WON = -1
    TIE = 0
    AGENT_WON = 1
    ONGOING = 2

def idx_to_action(int_action):
    if (int_action == actions.STAND.value):
        return actions.STAND
    
    if (int_action == actions.HIT.value):
        return actions.HIT
    
    if (int_action == actions.DOUBLE_DOWN.value):
        return actions.DOUBLE_DOWN
    
    if (int_action == actions.SPLIT.value):
        return actions.SPLIT

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