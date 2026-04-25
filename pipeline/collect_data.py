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


def play_vs_minimax_game(net, minimax_depth=3, board_size=15,
                         num_simulations=200, c_puct=1.5,
                         temperature_threshold=10):
    """
    Play a game between the RL agent (MCTS + net) and a MiniMax agent.
    Collect training data from BOTH sides:
      - RL turns: state + MCTS policy + game outcome
      - MiniMax turns: state + one-hot policy of MiniMax move + game outcome

    This teaches the network from MiniMax's "expert" moves, accelerating learning.

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

    # Randomly assign who goes first
    rl_player = 1 if np.random.random() < 0.5 else -1

    states = []
    policies = []
    players = []
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

            # Create one-hot policy from MiniMax's chosen move
            one_hot_policy = np.zeros(action_size)
            one_hot_policy[row * board.cols + col] = 1.0

            states.append(state)
            policies.append(one_hot_policy)
            players.append(current_player)

        board.make_move(row, col)
        move_count += 1

        game_result = board.get_game_ended()
        if game_result != 0:
            return _build_training_data(states, policies, players,
                                        game_result, board)

        if move_count >= board.rows * board.cols:
            break

    return _build_training_data(states, policies, players, 0.5, board)


def self_play_game(net, board_size=15, num_simulations=200,
                   c_puct=1.5, temperature_threshold=15):
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
