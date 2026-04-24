"""
Self-play data collection for AlphaZero-style training.

Runs games where the neural network plays against itself,
collecting (state, policy, value) training examples.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from mcts.mcts_alpha_zero import MCTS


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
        # Get state before move
        state = board.get_state_for_nn()

        # Choose temperature
        temp = 1.0 if move_count < temperature_threshold else 0.01

        # Get MCTS action probabilities
        action_probs = mcts.search(board, add_noise=True)

        # Store data
        states.append(state)
        policies.append(action_probs)
        players.append(board.turn)

        # Sample action
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
            # Invalid move - pick random valid move
            valid = board.get_valid_moves()
            if not valid:
                break
            row, col = valid[np.random.randint(len(valid))]

        board.make_move(row, col)
        move_count += 1

        # Check game end
        game_result = board.get_game_ended()
        if game_result != 0:
            # Assign values: +1 for winner, -1 for loser, 0 for draw
            training_data = []
            for i in range(len(states)):
                if game_result == 0.5:
                    value = 0.0
                else:
                    # game_result is +1 if originXO player won, -1 if other player won
                    if players[i] == board.originXO:
                        value = float(game_result)
                    else:
                        value = -float(game_result)

                training_data.append((states[i], policies[i], value))

                # Data augmentation: add symmetric versions
                symmetries = board.get_symmetries(states[i])
                action_size = board.rows * board.cols
                policy_2d = policies[i].reshape(board.rows, board.cols)

                for sym_state in symmetries[1:]:  # Skip original (already added)
                    # Determine the transformation applied
                    sym_policy = _transform_policy(policy_2d, sym_state, states[i],
                                                   board.rows, board.cols)
                    training_data.append((sym_state, sym_policy, value))

            return training_data

        # Safety check: max moves
        if move_count >= board.rows * board.cols:
            break

    # Draw (shouldn't normally reach here due to get_game_ended check)
    training_data = []
    for i in range(len(states)):
        training_data.append((states[i], policies[i], 0.0))
    return training_data


def _transform_policy(policy_2d, sym_state, orig_state, rows, cols):
    """
    Apply the same spatial transformation to the policy that was applied to the state.
    Uses rotation/flip matching to determine the correct transformation.
    """
    # Try all 8 transformations and find which one matches
    p = policy_2d.copy()

    # Generate all 8 transformations
    transforms = []
    # 0: original
    transforms.append(p)
    # 1: rot90
    transforms.append(np.rot90(p, 1))
    # 2: rot180
    transforms.append(np.rot90(p, 2))
    # 3: rot270
    transforms.append(np.rot90(p, 3))
    # 4: flip original
    transforms.append(np.fliplr(p))
    # 5: flip rot90
    transforms.append(np.fliplr(np.rot90(p, 1)))
    # 6: flip rot180
    transforms.append(np.fliplr(np.rot90(p, 2)))
    # 7: flip rot270
    transforms.append(np.fliplr(np.rot90(p, 3)))

    # Check which transformation of the original state matches sym_state
    orig_layer = orig_state[0]  # Use first layer for matching
    sym_layer = sym_state[0]

    orig_transforms = []
    orig_transforms.append(orig_layer)
    orig_transforms.append(np.rot90(orig_layer, 1))
    orig_transforms.append(np.rot90(orig_layer, 2))
    orig_transforms.append(np.rot90(orig_layer, 3))
    orig_transforms.append(np.fliplr(orig_layer))
    orig_transforms.append(np.fliplr(np.rot90(orig_layer, 1)))
    orig_transforms.append(np.fliplr(np.rot90(orig_layer, 2)))
    orig_transforms.append(np.fliplr(np.rot90(orig_layer, 3)))

    for idx, t in enumerate(orig_transforms):
        if np.array_equal(t, sym_layer):
            result = transforms[idx].flatten()
            result_sum = result.sum()
            if result_sum > 0:
                result /= result_sum
            return result

    # Fallback: return flattened original
    return policy_2d.flatten()


def collect_self_play_data(net, num_games=100, board_size=15,
                           num_simulations=200, verbose=True):
    """
    Collect training data from multiple self-play games.

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

    print("Starting self-play data collection...")
    data = collect_self_play_data(net, num_games=5, num_simulations=50)
    print(f"Collected {len(data)} training examples from 5 games")

    # Save sample data
    states = np.array([d[0] for d in data])
    policies = np.array([d[1] for d in data])
    values = np.array([d[2] for d in data])

    print(f"States shape: {states.shape}")
    print(f"Policies shape: {policies.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Value distribution: mean={values.mean():.3f}, std={values.std():.3f}")
