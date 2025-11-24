import online
import agents


def simulate_games(num_iterations, bj_game:online.game, agent:agents.RL_Agent, print_hands=False, learning=True):
    wins = 0.0
    losses = 0.0
    ties = 0.0

    while(num_iterations > 0):
        num_iterations -= 1
        if (num_iterations == 0 and print_hands):
            print("Debug statement")
        bj_game.new_game()
        
        result = 2
        states = []
        while(result == 2):
            state = (bj_game.get_player_cards(), bj_game.get_dealer_cards())

            if (learning):
                agent_action = agent.get_action_learning(state)
            else:
                agent_action = agent.get_action_trained(state)

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
    print("Simulation completed.")
    return (wins, ties, losses)

def print_game(wins, ties, losses):
    games_played = wins+ties+losses
    print("Results after ", games_played, " iterations.")
    print("Winrate: ",  f"{(wins / games_played) * 100:.2f}", "%", " | ",
          "Tierate: ",  f"{(ties / games_played) * 100:.2f}", "%", " | ",
          "Lossrate: ", f"{(losses / games_played) * 100:.2f}", "%")

def evaluate_agent(games_played, game:online.game, agent:agents.RL_Agent, train=True):
    if (train):
        _ = simulate_games(games_played, game, agent)

    wins, ties, losses = simulate_games(games_played, game, agent, learning=False)
    print_game(wins, ties, losses)
    

def main():
    games_played = 1000000
    #games_played = 100000

    game = online.game()

    random_agent = agents.Random_Agent()
    evaluate_agent(games_played, game, random_agent, train=False)

    naive_agent = agents.RL_Agent_Naive()
    evaluate_agent(games_played, game, naive_agent)

    value_agent = agents.RL_Agent_Naive_ValueBased()
    evaluate_agent(games_played, game, value_agent)


if __name__ == '__main__':
    main()