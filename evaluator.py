import online
import agents

num_games = 1000000

def main():
    print_hands = False
    num_iterations = num_games
    #num_iterations = 100
    games_played = num_iterations

    bj_game = online.game()
    #agent = RL_Agent_Naive()
    agent = agents.RL_Agent_Naive_ValueBased()
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