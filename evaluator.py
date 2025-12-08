import online
import agents
import helpers
import matplotlib.pyplot as plt
import numpy as np
import time

def simulate_games(num_iterations, bj_game:online.game, agent:agents.RL_Agent, print_hands=False, learning=True, graph=False, graph_attributes="", epoch_batch=5000):
    wins = 0.0
    losses = 0.0
    ties = 0.0

    #reset_iters = int(num_iterations/50)
    iters_elapsed = 0
    points = []

    while(num_iterations > 0):
        num_iterations -= 1
        if (num_iterations == 0 and print_hands):
            print("Debug statement")
        bj_game.new_game()
        
        result = helpers.result.ONGOING
        while(result == helpers.result.ONGOING):
            state = (bj_game.get_player_cards(), bj_game.get_dealer_cards())

            if (learning):
                agent_action = agent.get_action_learning(state)
            else:
                agent_action = agent.get_action_trained(state)

            result = bj_game.act(agent_action)

        agent.update_agent(result)
        agent_hand = bj_game.get_player_cards()
        dealer_hand = bj_game.get_dealer_cards()

        if (result == helpers.result.AGENT_WON):
            wins += 1
            if (print_hands):
                print("Won, player hand: ", agent_hand, " | dealer hand: ", dealer_hand)

        elif (result == helpers.result.TIE):
            ties += 1
            if (print_hands):
                print("Tied, player hand: ", agent_hand, " | dealer hand: ", dealer_hand)

        elif (result == helpers.result.DEALER_WON):
            losses += 1
            if (print_hands):
                print("Lost, player hand: ", agent_hand, " | dealer hand: ", dealer_hand)
        
        if(iters_elapsed == epoch_batch):
            # We will evaluate the agents current performance over a batch of games then plot those results
            wins_iter,ties_iter,losses_iter = simulate_games(epoch_batch,bj_game,agent,learning=False,graph=False)
            points.append((wins_iter,ties_iter,losses_iter))
            iters_elapsed = 0
        iters_elapsed += 1

    if (graph == True):
        data = np.array(points)
        wins = data[:, 0]
        ties = data[:, 1]
        losses = data[:, 2]

        win_rate = wins/epoch_batch
        tie_rate = ties/epoch_batch
        loss_rate = losses/epoch_batch

        x = np.arange(len(data)) * epoch_batch
        graph_modelType = graph_attributes
        plt.plot(x,win_rate, label="Win Rate")
        plt.plot(x,tie_rate, label="Tie Rate")
        plt.plot(x,win_rate+tie_rate, label="Win and Tie Rate")
        plt.plot(x,loss_rate, label="Loss Rate")
        plt.xlabel("Iterations")
        plt.ylabel("Rates(%)")
        plt.ylim(0,1)
        plt.title(graph_modelType)
        plt.grid(True)
        plt.legend()

    return (wins, ties, losses)

def print_game(wins, ties, losses):
    games_played = wins+ties+losses
    print("Results after ", games_played, " iterations.")
    print("Winrate: ",  f"{(wins / games_played) * 100:.2f}", "%", " | ",
          "Tierate: ",  f"{(ties / games_played) * 100:.2f}", "%", " | ",
          "Lossrate: ", f"{(losses / games_played) * 100:.2f}", "%")

def evaluate_agent(games_played, game:online.game, agent:agents.RL_Agent, train=True, graph_attributes="", batch_size=5000):
    end_time_train = 0
    start_time_train = time.time()
    if (train):
        _ = simulate_games(games_played, game, agent, learning=True, graph=True, graph_attributes=graph_attributes, epoch_batch=batch_size)
        end_time_train = time.time()
        plt.show()
    
    start_time_eval = time.time()
    wins, ties, losses = simulate_games(games_played, game, agent, learning=False)
    end_time_eval = time.time()
    print_game(wins, ties, losses)
    print("Total training time for ", games_played, f" iterations: {(end_time_train-start_time_train):.2f}s | Evaluation time: {(end_time_eval-start_time_eval):.2f}s")

def main():
    games_played = 1000000
    #games_played = 100000

    #game = online.game(bust_thresh=21,
    #                   hit_thresh=21)
    game = online.finite_deck_game(bust_thresh=21,
                                   hit_thresh=21,
                                   num_decks=6)

    random_agent = agents.Random_Agent()
    evaluate_agent(games_played, game, random_agent, train=False)

    naive_agent = agents.RL_Agent_Naive()
    evaluate_agent(games_played, game, naive_agent, graph_attributes="CardList Agent", batch_size=2500)

    value_agent = agents.RL_Agent_Naive_ValueBased()
    evaluate_agent(games_played, game, value_agent, graph_attributes="CardValues Agent", batch_size=2500)


if __name__ == '__main__':
    main()