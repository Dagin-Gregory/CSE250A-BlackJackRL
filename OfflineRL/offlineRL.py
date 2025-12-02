import pandas as pd
import numpy as np
import random
import ast


def parse_initial_hand(s):
    """
    Parse a string like "[10, 11]" into a list of ints [10, 11].
    """
    if not isinstance(s, str):
        return []
    s = s.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    if not s:
        return []
    parts = s.split(",")
    cards = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                cards.append(int(p))
            except ValueError:
                pass
    return cards


def hand_value(cards):
    """
    Compute blackjack total and whether the hand has a usable ace.
    Assumption: 11 represents an Ace.
    """
    if not cards:
        return 0, 0
    total = 0
    aces = 0
    for c in cards:
        if c == 11:
            aces += 1
            total += 11
        else:
            total += c
    usable_ace = 0
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    if aces > 0 and total <= 21:
        usable_ace = 1
    return total, usable_ace


def safe_eval(x):
    """
    Safely evaluate a string representation of a Python literal
    (list, nested list, etc.).
    """
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError, TypeError):
        return None


def encode_state(player_total, usable_ace, dealer_up):
    """
    Build a state tuple from raw features.
    State s = (player_total, usable_ace, dealer_up).
    """
    return int(player_total), int(usable_ace), int(dealer_up)


def build_state_index(states):
    """
    Build a mapping from state tuples to integer indices.
    Returns:
        state_to_id: dict mapping state tuple -> int
        id_to_state: list where index -> state tuple
    """
    state_to_id = {}
    id_to_state = []
    for s in states:
        if s not in state_to_id:
            idx = len(id_to_state)
            state_to_id[s] = idx
            id_to_state.append(s)
    return state_to_id, id_to_state


ACTIONS = ["H", "S"]


def encode_action(action_str):
    """
    Map action string to integer id.
    Actions:
        H -> 0
        S -> 1
    """
    mapping = {"H": 0, "S": 1}
    return mapping.get(action_str, -1)


def build_stepwise_tabular_data_from_raw_csv(
    csv_path="blackjack_simulator.csv",
    out_path="blackjack_rl_tabular_data.npz",
):
    """
    Build tabular RL data (state_ids, action_ids, rewards, id_to_state)
    from the raw blackjack_simulator.csv, using full action sequences
    and state s = (player_total, usable_ace, dealer_up).

    For each hand:
        - parse initial_hand (cards)
        - parse actions_taken (sequence of actions)
        - parse player_final (final cards)
        - skip hand if it ever uses D or P
        - reconstruct states BEFORE EACH ACTION (H / S)
        - assign the final reward 'win' to ALL steps in that hand

    This means a single played hand like:
        initial: 8 vs dealer 10
        actions: H, H, S
        finals: 21 and win = +1

    will generate three samples:
        (8, usable_ace?, 10),  H -> +1
        (11, usable_ace?, 10), H -> +1
        (21, usable_ace?, 10), S -> +1
    """
    df = pd.read_csv(csv_path)

    total_rows = len(df)
    print(f"[1/4] Building stepwise dataset from {total_rows} hands...")
    if total_rows == 0:
        print("No hands found in the CSV, aborting dataset construction.")
        return

    # log progress about every 5%
    progress_every = max(1, total_rows // 20)

    state_tuples = []
    action_ids = []
    rewards = []

    for idx, (_, row) in enumerate(df.iterrows()):
        initial_hand_str = row["initial_hand"]
        actions_str = row["actions_taken"]
        player_final_str = row["player_final"]
        dealer_up = row["dealer_up"]
        reward = float(row["win"])

        # Parse cards of the initial hand
        init_cards = parse_initial_hand(str(initial_hand_str))
        if len(init_cards) == 0:
            continue

        # Parse actions
        actions_raw = safe_eval(actions_str)
        if actions_raw is None:
            continue
        if isinstance(actions_raw, list) and len(actions_raw) > 0 and isinstance(actions_raw[0], list):
            actions_seq = actions_raw[0]
        else:
            actions_seq = actions_raw
        if not isinstance(actions_seq, list):
            continue
        actions_seq = [str(a) for a in actions_seq]

        # Skip if there are Double or Split actions
        if any(a in ["D", "P"] for a in actions_seq):
            continue
        # Only keep pure H/S sequences
        if not all(a in ["H", "S"] for a in actions_seq):
            continue

        # Parse final player hand (to reconstruct hit cards)
        pf_raw = safe_eval(player_final_str)
        if pf_raw is None or len(pf_raw) == 0:
            continue
        if isinstance(pf_raw[0], list):
            final_cards = pf_raw[0]
        else:
            final_cards = pf_raw
        if not isinstance(final_cards, list):
            continue

        # Number of hits must match number of new cards
        num_hits = actions_seq.count("H")
        if len(final_cards) < len(init_cards):
            continue
        if len(final_cards) != len(init_cards) + num_hits:
            continue

        # Cards drawn after the initial hand
        new_cards = final_cards[len(init_cards):]
        current_cards = list(init_cards)
        hit_index = 0

        # Reconstruct state -> action -> reward for each step
        for action in actions_seq:
            player_total, usable_ace = hand_value(current_cards)
            state = (int(player_total), int(usable_ace), int(dealer_up))

            a_id = encode_action(action)
            if a_id < 0:
                break

            state_tuples.append(state)
            action_ids.append(a_id)
            rewards.append(reward)  # Monte Carlo: final win/loss to every step

            # Apply the action to update the sequence of cards
            if action == "H":
                if hit_index >= len(new_cards):
                    # Inconsistent data, stop this hand
                    break
                current_cards.append(new_cards[hit_index])
                hit_index += 1

            # If player busts, the hand ends
            player_total, _ = hand_value(current_cards)
            if player_total > 21:
                break

        # progress
        if (idx + 1) % progress_every == 0 or (idx + 1) == total_rows:
            pct = (idx + 1) / total_rows * 100.0
            remaining = total_rows - (idx + 1)
            print(
                f"  Processed {idx + 1}/{total_rows} hands "
                f"({pct:.1f}%), ~{remaining} remaining"
            )

    # Build state index and arrays
    state_to_id, id_to_state = build_state_index(state_tuples)
    state_ids = np.array([state_to_id[s] for s in state_tuples], dtype=np.int32)
    action_ids = np.array(action_ids, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float64)

    np.savez(
        out_path,
        state_ids=state_ids,
        action_ids=action_ids,
        rewards=rewards,
        id_to_state=np.array(id_to_state, dtype=object),
    )

    print("Stepwise dataset built:")
    print("  num (state, action, reward) samples:", len(state_ids))
    print("  num unique states:", len(id_to_state))
    print("[1/4] Stepwise dataset construction finished.\n")


def load_tabular_data(path="blackjack_rl_tabular_data.npz"):
    """
    Load preprocessed tabular RL data from an .npz file.
    Returns state_ids, action_ids, rewards, id_to_state.
    """
    print("[2/4] Loading preprocessed tabular dataset...")
    data = np.load(path, allow_pickle=True)
    state_ids = data["state_ids"]
    action_ids = data["action_ids"]
    rewards = data["rewards"]
    id_to_state = data["id_to_state"]
    print("Dataset loaded successfully.\n")
    return state_ids, action_ids, rewards, id_to_state


def compute_tabular_q(state_ids, action_ids, rewards, num_states, num_actions):
    """
    Compute tabular Monte Carlo Q(s, a) as the average reward
    for each state-action pair observed in the dataset, with:

      1) Bayesian smoothing toward the global mean reward
      2) Conservative penalty, based only on the behavior data
    """
    print("[3/4] Computing tabular Q(s,a) with smoothing + conservative penalty...")

    q_sums = np.zeros((num_states, num_actions), dtype=np.float64)
    counts = np.zeros((num_states, num_actions), dtype=np.int64)

    for s, a, r in zip(state_ids, action_ids, rewards):
        if a < 0 or a >= num_actions:
            continue
        q_sums[s, a] += r
        counts[s, a] += 1

    # 1) Smoothing around the global mean reward
    global_mean = rewards.mean()
    alpha = 10.0  # pseudocount for smoothing Q
    q_values = np.zeros_like(q_sums)

    mask = counts > 0
    q_values[mask] = (q_sums[mask] + alpha * global_mean) / (counts[mask] + alpha)

    # 2) Conservative penalty (CQL-like): penalize less-behavioral actions
    lambda_cql = 0.1  # penalty strength (tunable)

    # sum of counts per state
    state_counts = counts.sum(axis=1, keepdims=True)  # shape (num_states, 1)
    # avoid divide-by-zero
    nonzero_states = state_counts.squeeze() > 0

    behavior_probs = np.zeros_like(q_values)
    behavior_probs[nonzero_states, :] = (
        counts[nonzero_states, :] / state_counts[nonzero_states, :]
    )

    # apply conservative penalty
    q_values = q_values - lambda_cql * (1.0 - behavior_probs)

    print(
        f"Q(s,a) computed with smoothing (alpha={alpha}, global_mean={global_mean:.3f}) "
        f"+ conservative penalty (lambda_cql={lambda_cql}).\n"
    )
    return q_values, counts


def derive_greedy_policy(q_values, counts):
    """
    Derive a greedy policy from Q(s, a) with a purely data-driven
    conservative rule:

        - Never choose actions with count(s,a) == 0 if there exists
          some action with count(s,a') > 0 in that state.
        - Among actions with data, use Q(s,a) + counts to choose the best.

    """
    print("Deriving greedy policy restricted to observed actions...")

    num_states, num_actions = q_values.shape
    policy_action_ids = np.zeros(num_states, dtype=np.int32)

    # hyperparameters (can be tuned)
    MIN_VISITS = 20       # if a_q has fewer visits than this, treat as weak
    SMALL_MARGIN = 0.05   # small Q difference => nearly tied

    for s in range(num_states):
        q_row = q_values[s]
        c_row = counts[s]

        total_visits = c_row.sum()
        if total_visits == 0:
            # state never seen in dataset: any choice is extrapolation
            # choose a neutral action (e.g., H = 0)
            policy_action_ids[s] = 0
            continue

        # actions that have at least 1 visit in this state
        valid_actions = np.where(c_row > 0)[0]

        if len(valid_actions) == 1:
            # Only one action observed in data -> never extrapolate
            policy_action_ids[s] = valid_actions[0]
            continue

        # More than one action with data -> use Q + counts
        # restricted to the valid subset
        q_valid = q_row[valid_actions]
        c_valid = c_row[valid_actions]

        # local index of the best action among valid ones
        local_best_idx = int(np.argmax(q_valid))
        a_q = int(valid_actions[local_best_idx])
        q_best = float(q_row[a_q])

        # second-best Q among valid actions
        q_second_best = float(
            np.max(np.delete(q_valid, local_best_idx))
        )

        # most frequent action among the valid subset
        local_freq_idx = int(np.argmax(c_valid))
        a_freq = int(valid_actions[local_freq_idx])

        # data-driven check: if a_q has few visits and the Q difference
        # from the second best is very small, prefer the more frequent action.
        if c_row[a_q] < MIN_VISITS and abs(q_best - q_second_best) < SMALL_MARGIN:
            policy_action_ids[s] = a_freq
        else:
            policy_action_ids[s] = a_q

    print("Greedy policy derived (using only actions supported by data).\n")
    return policy_action_ids


def save_q_and_policy(q_values, policy_action_ids, id_to_state, path="blackjack_tabular_q.npz"):
    """
    Save Q-table, greedy policy, states, and action names to an .npz file.
    """
    print("Saving Q-table and greedy policy to disk...")
    np.savez(
        path,
        q_values=q_values,
        policy_action_ids=policy_action_ids,
        id_to_state=id_to_state,
        actions=np.array(ACTIONS, dtype=object),
    )
    print(f"Model saved to {path}.\n")


def train_test_split(state_ids, action_ids, rewards, test_ratio=0.2, random_seed=42):
    """
    Split data into train and test sets by index.
    Returns:
        state_ids_train, action_ids_train, rewards_train,
        state_ids_test,  action_ids_test,  rewards_test
    """
    print("Performing train/test split...")
    n = len(state_ids)
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    state_ids_train = state_ids[train_idx]
    action_ids_train = action_ids[train_idx]
    rewards_train = rewards[train_idx]
    state_ids_test = state_ids[test_idx]
    action_ids_test = action_ids[test_idx]
    rewards_test = rewards[test_idx]
    print(f"Split done: {len(state_ids_train)} train / {len(state_ids_test)} test.\n")
    return (
        state_ids_train,
        action_ids_train,
        rewards_train,
        state_ids_test,
        action_ids_test,
        rewards_test,
    )


def evaluate_policy_accuracy_on_test(state_ids_test, action_ids_test, policy_action_ids):
    """
    Evaluate how often the greedy policy chooses the same action
    as the behavior policy in the test set.
    Returns:
        accuracy (fraction of correct predictions)
    """
    total = len(state_ids_test)
    if total == 0:
        return 0.0
    correct = 0
    for s, a_beh in zip(state_ids_test, action_ids_test):
        a_pred = policy_action_ids[s]
        if a_pred == a_beh:
            correct += 1
    accuracy = correct / total
    return accuracy


def compute_outcome_rates(rewards):
    """
    Compute win, tie, and loss rates given an array of rewards.
    Assumes:
        reward > 0 -> win
        reward = 0 -> tie
        reward < 0 -> loss
    Returns:
        win_rate, tie_rate, loss_rate, n_win, n_tie, n_loss
    """
    n = len(rewards)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0
    rewards = np.asarray(rewards)
    wins = rewards > 0
    ties = rewards == 0
    losses = rewards < 0
    n_win = int(wins.sum())
    n_tie = int(ties.sum())
    n_loss = int(losses.sum())
    win_rate = n_win / n
    tie_rate = n_tie / n
    loss_rate = n_loss / n
    return win_rate, tie_rate, loss_rate, n_win, n_tie, n_loss


def load_q_model(path="blackjack_tabular_q.npz"):
    """
    Load Q-table, greedy policy, and state mapping from disk.
    """
    print("Loading Q-table and greedy policy from disk...")
    data = np.load(path, allow_pickle=True)
    q_values = data["q_values"]
    policy_action_ids = data["policy_action_ids"]
    id_to_state = data["id_to_state"]
    actions = data["actions"]
    print("Model loaded.\n")
    return q_values, policy_action_ids, id_to_state, actions


def build_state_to_id_from_id_to_state(id_to_state):
    """
    Build a dictionary mapping state tuple -> state_id
    from the id_to_state array.
    """
    state_to_id = {}
    for idx, s in enumerate(id_to_state):
        state_to_id[tuple(s)] = idx
    return state_to_id


def greedy_action_for_state(state, policy_action_ids, id_to_state, actions=ACTIONS, state_to_id=None):
    """
    Given a raw state (player_total, usable_ace, dealer_up),
    return the greedy action name, action id, and state id.
    """
    if state_to_id is None:
        state_to_id = build_state_to_id_from_id_to_state(id_to_state)
    state_tuple = tuple(int(x) for x in state)
    if state_tuple not in state_to_id:
        return None, None, None
    sid = state_to_id[state_tuple]
    action_id = policy_action_ids[sid]
    action_name = actions[action_id]
    return action_name, action_id, sid


class SimpleBlackjackSimulator:
    """
    Simple multi-step blackjack simulator suitable for testing the learned policy.

    Uses a finite multi-deck shoe (casino style) with reshuffle at 25% remaining.
    State used by RL:
        (player_total, usable_ace, dealer_up)

    Behavior:
        - Player can choose H or S multiple times.
        - We simulate step-by-step until player Stands or busts.
        - Dealer then plays according to rules when player Stands.
    """

    def __init__(self, num_decks=2, penetration=0.25, hit_soft_17=True, seed=None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.hit_soft_17 = hit_soft_17
        if seed is not None:
            random.seed(seed)
        self._init_shoe()

    def _init_shoe(self):
        """
        Initialize and shuffle a fresh shoe with num_decks standard decks.
        Card values:
            2-9 as themselves
            10, J, Q, K as 10
            A as 11
        """
        ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        shoe = ranks * 4 * self.num_decks
        random.shuffle(shoe)
        self.shoe = shoe
        self.total_cards = len(self.shoe)
        self.shuffle_cutoff = int(self.total_cards * self.penetration)

    def draw_card(self):
        """
        Draw the top card from the shoe.
        If the shoe is at or below the remaining threshold, reshuffle.
        """
        if len(self.shoe) <= self.shuffle_cutoff:
            self._init_shoe()
        return self.shoe.pop()

    def reset(self):
        """
        Deal initial player and dealer hands and return the RL state.
        State = (player_total, usable_ace, dealer_up).

        Also sets internal flags for natural blackjacks.
        """
        self.player_cards = [self.draw_card(), self.draw_card()]
        self.dealer_cards = [self.draw_card(), self.draw_card()]

        player_total, usable_ace = hand_value(self.player_cards)
        dealer_up = self.dealer_cards[0]

        # Naturals
        self.initial_natural = (player_total == 21 and len(self.player_cards) == 2)
        dealer_total, _ = hand_value(self.dealer_cards)
        self.dealer_natural = (dealer_total == 21 and len(self.dealer_cards) == 2)

        state = (player_total, usable_ace, dealer_up)
        return state

    def dealer_play(self):
        """
        Play out the dealer hand according to standard rules:
        hit until 17+, optionally hit soft 17 depending on configuration.
        """
        while True:
            total, usable_ace = hand_value(self.dealer_cards)
            if total > 21:
                break
            if total < 17:
                self.dealer_cards.append(self.draw_card())
                continue
            if total == 17 and self.hit_soft_17 and usable_ace == 1:
                self.dealer_cards.append(self.draw_card())
                continue
            break

    def step(self, action_name):
        """
        Execute ONE player decision step:
            H: Hit        -> draw a card, possibly bust
            S: Stand      -> dealer plays, hand resolves

        Returns:
            next_state: tuple or None
            reward: float (e.g., -1, 0, 1, 1.5)
            done: bool (True if the hand ended)
        """

        # If player has a natural blackjack, resolve immediately
        if self.initial_natural:
            if self.dealer_natural:
                return None, 0.0, True
            return None, 1.5, True

        if action_name not in ("H", "S"):
            raise ValueError(f"Invalid action: {action_name}")

        if action_name == "H":
            # Player takes a card
            self.player_cards.append(self.draw_card())
            player_total, usable_ace = hand_value(self.player_cards)
            dealer_up = self.dealer_cards[0]

            # Bust -> immediate loss
            if player_total > 21:
                return None, -1.0, True

            # Otherwise, hand continues, reward is 0 for this step
            next_state = (player_total, usable_ace, dealer_up)
            return next_state, 0.0, False

        # action_name == "S": player stands, dealer plays out
        self.dealer_play()
        dealer_total, _ = hand_value(self.dealer_cards)
        player_total, _ = hand_value(self.player_cards)

        if dealer_total > 21:
            reward = 1.0
        elif player_total > dealer_total:
            reward = 1.0
        elif player_total < dealer_total:
            reward = -1.0
        else:
            reward = 0.0

        return None, reward, True


def test_on_new_simulator(simulator, num_episodes=100000, q_model_path="blackjack_tabular_q.npz"):
    """
    Uses the learned policy to play blackjack on a simulator in a
    multi-step way:

        state = simulator.reset()
        while not done:
            a = argmax_a Q(s, a)
            s, r, done = simulator.step(a)

    The simulator object must implement:
        reset() -> initial_state
        step(action_name: str) -> (next_state, reward, done)

    Returns a dictionary with win/tie/loss rates.
    """
    q_values, policy_action_ids, id_to_state, actions = load_q_model(q_model_path)
    state_to_id = build_state_to_id_from_id_to_state(id_to_state)

    wins = 0
    ties = 0
    losses = 0

    print("[4/4] Evaluating greedy policy on the simulator (online)...")
    print(f"Simulating {num_episodes} episodes.")
    progress_every = max(1, num_episodes // 10)  # log every 10%

    for i in range(num_episodes):
        state = simulator.reset()
        done = False
        final_reward = 0.0

        while not done:
            # Choose greedy action; if state unseen, fallback to random
            action_name, action_id, sid = greedy_action_for_state(
                state, policy_action_ids, id_to_state, actions=actions, state_to_id=state_to_id
            )
            if action_name is None:
                action_name = random.choice(list(actions))

            next_state, reward, done = simulator.step(action_name)
            if done:
                final_reward = reward
            state = next_state

        if final_reward > 0:
            wins += 1
        elif final_reward == 0:
            ties += 1
        else:
            losses += 1

        if (i + 1) % progress_every == 0 or (i + 1) == num_episodes:
            pct = (i + 1) / num_episodes * 100.0
            remaining = num_episodes - (i + 1)
            print(
                f"  Episodes simulated: {i + 1}/{num_episodes} "
                f"({pct:.1f}%), ~{remaining} remaining"
            )

    total = num_episodes
    win_rate = wins / total
    tie_rate = ties / total
    loss_rate = losses / total
    print("[4/4] Simulator evaluation finished.\n")
    return {
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "loss_rate": loss_rate,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def main():
    
    # 1) Build stepwise dataset from raw CSV, using ALL actions (H/S) per hand
    build_stepwise_tabular_data_from_raw_csv(
        csv_path="blackjack_simulator.csv",
        out_path="blackjack_rl_tabular_data.npz",
    )
    
    # 2) Load tabular data
    state_ids, action_ids, rewards, id_to_state = load_tabular_data("blackjack_rl_tabular_data.npz")
    num_states = len(id_to_state)
    num_actions = len(ACTIONS)

    # 3) Train/test split
    (
        state_ids_train,
        action_ids_train,
        rewards_train,
        state_ids_test,
        action_ids_test,
        rewards_test,
    ) = train_test_split(
        state_ids=state_ids,
        action_ids=action_ids,
        rewards=rewards,
        test_ratio=0.2,
        random_seed=42,
    )

    print("Total samples:", len(state_ids))
    print("Train samples:", len(state_ids_train))
    print("Test  samples:", len(state_ids_test))
    print()

    # 4) Monte Carlo Q-learning (offline) with ALL steps
    q_values, counts = compute_tabular_q(
        state_ids=state_ids_train,
        action_ids=action_ids_train,
        rewards=rewards_train,
        num_states=num_states,
        num_actions=num_actions,
    )

    # 5) Derive greedy policy
    policy_action_ids = derive_greedy_policy(q_values, counts)

    # 6) Save model
    save_q_and_policy(
        q_values=q_values,
        policy_action_ids=policy_action_ids,
        id_to_state=id_to_state,
        path="blackjack_tabular_q.npz",
    )

    # 7) Offline evaluation on held-out dataset
    print("=== Offline evaluation (train/test) ===")
    accuracy = evaluate_policy_accuracy_on_test(
        state_ids_test=state_ids_test,
        action_ids_test=action_ids_test,
        policy_action_ids=policy_action_ids,
    )
    error = 1.0 - accuracy

    win_b, tie_b, loss_b, n_wb, n_tb, n_lb = compute_outcome_rates(rewards_test)

    same_action_mask = np.array([
        policy_action_ids[s] == a_beh
        for s, a_beh in zip(state_ids_test, action_ids_test)
    ])
    rewards_greedy_eval = rewards_test[same_action_mask]
    win_g, tie_g, loss_g, n_wg, n_tg, n_lg = compute_outcome_rates(rewards_greedy_eval)
    coverage = len(rewards_greedy_eval) / len(rewards_test) if len(rewards_test) > 0 else 0.0

    print(f"Accuracy (same action as behavior): {accuracy * 100:.2f}%")
    print(f"Error (different action):           {error * 100:.2f}%")

    print("\n--- Behavior policy outcomes on TEST ---")
    print(f"Win rate:  {win_b * 100:.2f}%  (wins:  {n_wb})")
    print(f"Tie rate:  {tie_b * 100:.2f}%  (ties:  {n_tb})")
    print(f"Loss rate: {loss_b * 100:.2f}%  (losses:{n_lb})")

    print("\n--- Greedy policy outcomes on TEST (where action matches behavior) ---")
    print(f"Coverage:  {coverage * 100:.2f}% of the test set")
    print(f"Win rate:  {win_g * 100:.2f}%  (wins:  {n_wg})")
    print(f"Tie rate:  {tie_g * 100:.2f}%  (ties:  {n_tg})")
    print(f"Loss rate: {loss_g * 100:.2f}%  (losses:{n_lg})")

    print("\nExample of first 10 states and greedy actions:")
    for idx in range(min(10, num_states)):
        state = id_to_state[idx]
        a_star = policy_action_ids[idx]
        action_name = ACTIONS[a_star]
        print(f"State {idx}: {state} -> greedy action: {action_name}, Q = {q_values[idx, a_star]:.3f}")

    # 8) Online evaluation on the improved multi-step simulator
    simulator = SimpleBlackjackSimulator()
    sim_results = test_on_new_simulator(simulator, num_episodes=1000000, q_model_path="blackjack_tabular_q.npz")
    print("\n=== Greedy policy evaluated on SimpleBlackjackSimulator (1M hands) ===")
    print(f"Win rate:  {sim_results['win_rate'] * 100:.2f}%  (wins:  {sim_results['wins']})")
    print(f"Tie rate:  {sim_results['tie_rate'] * 100:.2f}%  (ties:  {sim_results['ties']})")
    print(f"Loss rate: {sim_results['loss_rate'] * 100:.2f}%  (losses:{sim_results['losses']})")

   

if __name__ == "__main__":
    main()
