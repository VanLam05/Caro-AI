"""
Data collection for AlphaZero-style training.

Supports two modes:
  1. vs MiniMax: RL agent plays against MiniMax agent (faster learning)
  2. Self-play: RL agent plays against itself (for fine-tuning)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from mcts.mcts_alpha_zero import MCTS
from models.agentMiniMax import AgentMiniMax


def _flip_state(state):
    """
    Flip board perspective: swap current player and opponent layers.
    Layer 0 (my pieces) <-> Layer 1 (opponent pieces).
    Layers 2-3 stay the same.
    """
    flipped = state.copy()
    flipped[0], flipped[1] = state[1].copy(), state[0].copy()
    return flipped


def play_vs_minimax_game(net, minimax_depth=3, board_size=15,
                         num_simulations=100, c_puct=2.5,
                         temperature_threshold=10,
                         label_smoothing=0.1):
    """
    Play a game between the RL agent (MCTS + net) and a MiniMax agent.
    Collect training data from BOTH sides:
      - RL turns: state + MCTS policy + game outcome
      - MiniMax turns: state + one-hot policy of MiniMax move + game outcome

    When MiniMax wins, also generates perspective-flipped data so RL
    can learn MiniMax's winning strategy from its own perspective.

    Args:
        net: GomokuNet neural network
        minimax_depth: depth of the MiniMax opponent
        board_size: board dimensions
        num_simulations: MCTS simulations per move
        c_puct: exploration constant
        temperature_threshold: use temperature=1 for first N moves

    Returns:
        list of (state, policy, value) tuples
    """
    board = Board(rows=board_size, cols=board_size, winning_condition=5)
    mcts = MCTS(net, num_simulations=num_simulations, c_puct=c_puct)
    minimax = AgentMiniMax(board, max_depth=minimax_depth)

    # RL always plays as player 1 (goes first).
    # MiniMax must be player -1 because its heuristic is hardcoded for -1.
    rl_player = 1

    states = []
    policies = []
    players = []

    # Track MiniMax moves separately for perspective flipping
    minimax_states = []
    minimax_policies = []

    action_size = board_size * board_size

    move_count = 0

    while True:
        state = board.get_state_for_nn()
        current_player = board.turn

        if current_player == rl_player:
            # RL agent's turn: use MCTS
            temp = 1.0 if move_count < temperature_threshold else 0.01
            action_probs = mcts.search(board, add_noise=True)

            states.append(state)
            policies.append(action_probs)
            players.append(current_player)

            if temp < 0.1:
                action = np.argmax(action_probs)
            else:
                probs = action_probs ** (1.0 / temp)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                    action = np.random.choice(len(probs), p=probs)
                else:
                    valid = board.get_valid_moves()
                    if not valid:
                        break
                    r, c = valid[np.random.randint(len(valid))]
                    action = r * board.cols + c

            row = action // board.cols
            col = action % board.cols

            if board.grid[row][col] != 0:
                valid = board.get_valid_moves()
                if not valid:
                    break
                row, col = valid[np.random.randint(len(valid))]
        else:
            # MiniMax's turn: use MiniMax agent, create one-hot policy as target
            move = minimax.get_move()
            if move is None:
                break
            row, col = move

            # Create smoothed policy from MiniMax's chosen move
            # Label smoothing distributes small probability to other valid moves
            valid = board.get_valid_moves()
            one_hot_policy = np.zeros(action_size)
            if label_smoothing > 0 and len(valid) > 1:
                smooth_prob = label_smoothing / len(valid)
                for vr, vc in valid:
                    one_hot_policy[vr * board.cols + vc] = smooth_prob
                one_hot_policy[row * board.cols + col] = (
                    1.0 - label_smoothing + smooth_prob
                )
            else:
                one_hot_policy[row * board.cols + col] = 1.0

            states.append(state)
            policies.append(one_hot_policy)
            players.append(current_player)

            # Save MiniMax state+policy for perspective flipping
            minimax_states.append(state)
            minimax_policies.append(one_hot_policy)

        board.make_move(row, col)
        move_count += 1

        game_result = board.get_game_ended()
        if game_result != 0:
            data = _build_training_data(states, policies, players,
                                        game_result, board)

            # If MiniMax won, flip its winning moves to teach RL
            minimax_player = -rl_player
            minimax_won = (
                (game_result == 1 and minimax_player == board.originXO) or
                (game_result == -1 and minimax_player != board.originXO)
            )
            if minimax_won and minimax_states:
                flipped = _build_flipped_data(
                    minimax_states, minimax_policies, board
                )
                data.extend(flipped)

            return data

        if move_count >= board.rows * board.cols:
            break

    return _build_training_data(states, policies, players, 0.5, board)


def _build_flipped_data(minimax_states, minimax_policies, board):
    """
    Create training data from MiniMax's perspective flipped to RL's.

    When MiniMax wins, its moves represent a winning strategy.
    By flipping the board (swapping player/opponent layers), we create
    examples that teach RL: "in this equivalent position, play this move
    to win".

    The flipped examples have value=+1 (winning) because MiniMax won
    using these moves.
    """
    flipped_data = []

    for state, policy in zip(minimax_states, minimax_policies):
        # Flip perspective: swap layer 0 (my pieces) and layer 1 (opponent)
        flipped_state = _flip_state(state)
        # Policy stays the same (same move, just from other player's view)
        # Value = +1 because this move led to winning
        flipped_data.append((flipped_state, policy, 1.0))

        # Also add augmented versions
        symmetries = board.get_symmetries(flipped_state)
        policy_2d = policy.reshape(board.rows, board.cols)

        for sym_state in symmetries[1:]:
            sym_policy = _transform_policy(
                policy_2d, sym_state, flipped_state,
                board.rows, board.cols
            )
            flipped_data.append((sym_state, sym_policy, 1.0))

    return flipped_data


def self_play_game(net, board_size=15, num_simulations=200,
                   c_puct=2.5, temperature_threshold=15):
    """
    Play a single self-play game and collect training data.

    Args:
        net: GomokuNet neural network
        board_size: size of the board
        num_simulations: MCTS simulations per move
        c_puct: exploration constant
        temperature_threshold: use temperature=1 for first N moves, then 0

    Returns:
        list of (state, policy, value) tuples for training
    """
    board = Board(rows=board_size, cols=board_size, winning_condition=5)
    mcts = MCTS(net, num_simulations=num_simulations, c_puct=c_puct)

    states = []
    policies = []
    players = []

    move_count = 0

    while True:
        state = board.get_state_for_nn()
        temp = 1.0 if move_count < temperature_threshold else 0.01
        action_probs = mcts.search(board, add_noise=True)

        states.append(state)
        policies.append(action_probs)
        players.append(board.turn)

        if temp < 0.1:
            action = np.argmax(action_probs)
        else:
            probs = action_probs ** (1.0 / temp)
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
                action = np.random.choice(len(probs), p=probs)
            else:
                valid = board.get_valid_moves()
                if not valid:
                    break
                r, c = valid[np.random.randint(len(valid))]
                action = r * board.cols + c

        row = action // board.cols
        col = action % board.cols

        if board.grid[row][col] != 0:
            valid = board.get_valid_moves()
            if not valid:
                break
            row, col = valid[np.random.randint(len(valid))]

        board.make_move(row, col)
        move_count += 1

        game_result = board.get_game_ended()
        if game_result != 0:
            return _build_training_data(states, policies, players,
                                        game_result, board)

        if move_count >= board.rows * board.cols:
            break

    return _build_training_data(states, policies, players, 0.5, board)


def _build_training_data(states, policies, players, game_result, board):
    """
    Build training examples from a completed game with data augmentation.

    Args:
        states: list of board states
        policies: list of policy distributions
        players: list of which player was to move
        game_result: game outcome (1, -1, or 0.5 for draw)
        board: Board object (for symmetry generation)

    Returns:
        list of (state, policy, value) tuples
    """
    training_data = []
    action_size = board.rows * board.cols

    for i in range(len(states)):
        if game_result == 0.5:
            value = 0.0
        else:
            if players[i] == board.originXO:
                value = float(game_result)
            else:
                value = -float(game_result)

        training_data.append((states[i], policies[i], value))

        # Data augmentation: rotations and flips
        symmetries = board.get_symmetries(states[i])
        policy_2d = policies[i].reshape(board.rows, board.cols)

        for sym_state in symmetries[1:]:
            sym_policy = _transform_policy(policy_2d, sym_state, states[i],
                                           board.rows, board.cols)
            training_data.append((sym_state, sym_policy, value))

    return training_data


def _transform_policy(policy_2d, sym_state, orig_state, rows, cols):
    """Apply the same spatial transformation to the policy as was applied to the state."""
    p = policy_2d.copy()

    transforms = [
        p,
        np.rot90(p, 1),
        np.rot90(p, 2),
        np.rot90(p, 3),
        np.fliplr(p),
        np.fliplr(np.rot90(p, 1)),
        np.fliplr(np.rot90(p, 2)),
        np.fliplr(np.rot90(p, 3)),
    ]

    orig_layer = orig_state[0]
    sym_layer = sym_state[0]

    orig_transforms = [
        orig_layer,
        np.rot90(orig_layer, 1),
        np.rot90(orig_layer, 2),
        np.rot90(orig_layer, 3),
        np.fliplr(orig_layer),
        np.fliplr(np.rot90(orig_layer, 1)),
        np.fliplr(np.rot90(orig_layer, 2)),
        np.fliplr(np.rot90(orig_layer, 3)),
    ]

    for idx, t in enumerate(orig_transforms):
        if np.array_equal(t, sym_layer):
            result = transforms[idx].flatten()
            result_sum = result.sum()
            if result_sum > 0:
                result /= result_sum
            return result

    return policy_2d.flatten()


def collect_vs_minimax_data(net, num_games=50, minimax_depth=3,
                            board_size=15, num_simulations=200, verbose=True):
    """
    Collect training data from games against MiniMax.

    Args:
        net: GomokuNet neural network
        num_games: number of games to play
        minimax_depth: MiniMax search depth
        board_size: board dimensions
        num_simulations: MCTS simulations per move
        verbose: print progress

    Returns:
        list of (state, policy, value) tuples
    """
    all_data = []

    for game_idx in range(num_games):
        game_data = play_vs_minimax_game(
            net, minimax_depth=minimax_depth,
            board_size=board_size,
            num_simulations=num_simulations,
        )
        all_data.extend(game_data)

        if verbose and (game_idx + 1) % 5 == 0:
            print(f"  vs MiniMax (depth={minimax_depth}): "
                  f"{game_idx + 1}/{num_games} games, "
                  f"{len(all_data)} examples collected")

    return all_data


def collect_self_play_data(net, num_games=100, board_size=15,
                           num_simulations=200, verbose=True):
    """
    Collect training data from self-play games.

    Args:
        net: GomokuNet neural network
        num_games: number of self-play games to play
        board_size: size of the board
        num_simulations: MCTS simulations per move
        verbose: print progress

    Returns:
        list of (state, policy, value) tuples
    """
    all_data = []

    for game_idx in range(num_games):
        game_data = self_play_game(
            net, board_size=board_size,
            num_simulations=num_simulations,
        )
        all_data.extend(game_data)

        if verbose and (game_idx + 1) % 5 == 0:
            print(f"  Self-play: {game_idx + 1}/{num_games} games, "
                  f"{len(all_data)} training examples collected")

    return all_data


def generate_tactical_data(board_size=15, num_examples=500):
    """
    Generate synthetic training examples for winning/blocking patterns.

    Creates board positions with 4-in-a-row and teaches the network
    to play the winning 5th move (or block opponent's 5th).

    Returns:
        list of (state, policy, value) tuples
    """
    data = []
    action_size = board_size * board_size
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for _ in range(num_examples):
        board = Board(rows=board_size, cols=board_size, winning_condition=5)
        player = board.turn
        direction = directions[np.random.randint(len(directions))]
        dr, dc = direction

        # Random starting position (ensure 5 cells fit)
        max_r = board_size - 1 - max(0, 4 * dr)
        max_c = board_size - 1 - max(0, 4 * dc)
        min_r = max(0, -4 * dr)
        min_c = max(0, -4 * dc)

        if min_r > max_r or min_c > max_c:
            continue

        start_r = np.random.randint(min_r, max_r + 1)
        start_c = np.random.randint(min_c, max_c + 1)

        # Place 4 pieces, leave gap at random position for the 5th
        gap_idx = np.random.randint(5)
        positions = []
        for k in range(5):
            r = start_r + k * dr
            c = start_c + k * dc
            positions.append((r, c))

        # Check all positions are valid
        valid = all(0 <= r < board_size and 0 <= c < board_size
                    for r, c in positions)
        if not valid:
            continue

        # Place 4 of 5 pieces for current player
        for k in range(5):
            if k == gap_idx:
                continue
            r, c = positions[k]
            board.grid[r][c] = player

        # Add some random opponent pieces for context (avoid gap position)
        gap_pos = positions[gap_idx]
        num_opp = np.random.randint(3, 8)
        for _ in range(num_opp):
            er = np.random.randint(board_size)
            ec = np.random.randint(board_size)
            if board.grid[er][ec] == 0 and (er, ec) != gap_pos:
                board.grid[er][ec] = -player
                board.move_history.append((er, ec, -player))

        # Add current player's pieces to history
        for k in range(5):
            if k != gap_idx:
                r, c = positions[k]
                board.move_history.append((r, c, player))

        # Set turn to current player
        board.turn = player

        # The winning move is at the gap position
        win_r, win_c = positions[gap_idx]
        state = board.get_state_for_nn()

        # Policy: one-hot on winning move
        policy = np.zeros(action_size)
        policy[win_r * board_size + win_c] = 1.0

        # Value: +1 (winning position)
        value = 1.0

        data.append((state, policy, value))

        # Also generate blocking version: opponent's perspective
        # Flip board: swap player pieces
        board2 = Board(rows=board_size, cols=board_size, winning_condition=5)
        for i in range(board_size):
            for j in range(board_size):
                if board.grid[i][j] != 0:
                    board2.grid[i][j] = -board.grid[i][j]
        board2.turn = -player
        board2.move_history = [(r, c, -p) for r, c, p in board.move_history]

        state2 = board2.get_state_for_nn()
        # Policy: must block the same position
        policy2 = np.zeros(action_size)
        policy2[win_r * board_size + win_c] = 1.0
        # Value: -1 (losing if don't block, but blocking saves)
        value2 = 0.0

        data.append((state2, policy2, value2))

    return data


if __name__ == '__main__':
    import torch
    from neural_net.architecture import GomokuNet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    net = GomokuNet(board_size=15, in_channels=4, num_res_blocks=5, channels=128)
    net.to(device)

    print("\n--- Testing vs MiniMax data collection ---")
    data = collect_vs_minimax_data(net, num_games=5, minimax_depth=2,
                                   num_simulations=50)
    print(f"Collected {len(data)} training examples from 5 games vs MiniMax")

    print("\n--- Testing self-play data collection ---")
    data2 = collect_self_play_data(net, num_games=5, num_simulations=50)
    print(f"Collected {len(data2)} training examples from 5 self-play games")

    print("\n--- Testing tactical data generation ---")
    data3 = generate_tactical_data(num_examples=100)
    print(f"Generated {len(data3)} tactical training examples")
