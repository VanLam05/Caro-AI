import math
import numpy as np


class MCTSNode:
    """A node in the Monte Carlo search tree."""

    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, c_puct):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_parent = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            ucb = child.q_value + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors):
        """
        Expand node with given action-prior pairs.

        Args:
            action_priors: list of (action, prior_probability) tuples
        """
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def backpropagate(self, value):
        """Update this node and all ancestors with the evaluation value."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            # Negate value because parent is opponent's perspective
            self.parent.backpropagate(-value)


class MCTS:
    """
    Monte Carlo Tree Search guided by a neural network (AlphaZero style).

    The neural network provides:
      - policy: prior probabilities for each action
      - value: estimated outcome from the current position
    """

    def __init__(self, neural_net, num_simulations=200, c_puct=2.5,
                 dirichlet_alpha=0.15, dirichlet_epsilon=0.25):
        self.net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, board, add_noise=True):
        """
        Run MCTS simulations from the given board state.

        Args:
            board: Board object (will be copied internally)
            add_noise: whether to add Dirichlet noise at root (for exploration during training)

        Returns:
            action_probs: numpy array of shape (225,) with visit-count-based probabilities
        """
        root = MCTSNode()
        state = board.get_state_for_nn()
        policy, value = self.net.predict(state)

        valid_moves = board.get_valid_moves_optimized()
        if not valid_moves:
            return np.zeros(board.rows * board.cols)

        # Mask invalid moves and re-normalize
        action_size = board.rows * board.cols
        mask = np.zeros(action_size)
        for r, c in valid_moves:
            mask[r * board.cols + c] = 1.0

        policy = policy * mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # Uniform over valid moves
            policy = mask / mask.sum()

        # Add Dirichlet noise at root for exploration
        if add_noise and len(valid_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
            noise_full = np.zeros(action_size)
            for idx, (r, c) in enumerate(valid_moves):
                noise_full[r * board.cols + c] = noise[idx]
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise_full

        # Expand root
        action_priors = []
        for r, c in valid_moves:
            action = r * board.cols + c
            action_priors.append((action, policy[action]))
        root.expand(action_priors)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = board.copy()

            # Selection: traverse tree using UCB
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                r, c = action // sim_board.cols, action % sim_board.cols
                sim_board.make_move(r, c)

            # Check if game ended
            game_result = sim_board.get_game_ended()
            if game_result != 0:
                # Terminal node
                if game_result == 0.5:
                    leaf_value = 0.0  # Draw
                else:
                    # game_result is 1 or -1 relative to originXO
                    # We need value relative to the player who just moved (parent of this node)
                    # The current sim_board.turn is the player who is about to move
                    # So the player who just moved is -sim_board.turn
                    last_player = -sim_board.turn
                    if last_player == sim_board.originXO:
                        leaf_value = game_result
                    else:
                        leaf_value = -game_result
                    # Negate because backprop expects value from current node's perspective
                    leaf_value = -leaf_value
            else:
                # Evaluate with neural network
                state = sim_board.get_state_for_nn()
                leaf_policy, leaf_value = self.net.predict(state)

                sim_valid = sim_board.get_valid_moves_optimized()
                if sim_valid:
                    mask = np.zeros(action_size)
                    for r, c in sim_valid:
                        mask[r * sim_board.cols + c] = 1.0
                    leaf_policy = leaf_policy * mask
                    lp_sum = leaf_policy.sum()
                    if lp_sum > 0:
                        leaf_policy /= lp_sum
                    else:
                        leaf_policy = mask / mask.sum()

                    action_priors = []
                    for r, c in sim_valid:
                        a = r * sim_board.cols + c
                        action_priors.append((a, leaf_policy[a]))
                    node.expand(action_priors)

                # Value from network is from current player's perspective
                # Negate for backpropagation (parent is opponent)
                leaf_value = -leaf_value

            node.backpropagate(leaf_value)

        # Build action probability distribution from visit counts
        action_probs = np.zeros(action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total_visits = action_probs.sum()
        if total_visits > 0:
            action_probs /= total_visits

        return action_probs

    def get_action(self, board, temperature=0.0):
        """
        Get action from MCTS search with tactical override.

        Checks for immediate wins/blocks before running MCTS.

        Args:
            board: Board object
            temperature: controls exploration.
                0 = pick best move (greedy)
                1 = sample proportional to visit counts
                >0 = sample with temperature scaling

        Returns:
            action: (row, col) tuple
            action_probs: numpy array of shape (225,)
        """
        action_size = board.rows * board.cols

        # Tactical override: check immediate win
        current_player = board.turn
        win_move = board.find_winning_move(current_player)
        if win_move is not None:
            action_probs = np.zeros(action_size)
            r, c = win_move
            action_probs[r * board.cols + c] = 1.0
            return win_move, action_probs

        # Tactical override: block opponent's immediate win
        opponent = -current_player
        block_move = board.find_winning_move(opponent)
        if block_move is not None:
            action_probs = np.zeros(action_size)
            r, c = block_move
            action_probs[r * board.cols + c] = 1.0
            return block_move, action_probs

        action_probs = self.search(board, add_noise=(temperature > 0))

        if temperature == 0:
            # Greedy
            action = np.argmax(action_probs)
        else:
            # Sample with temperature
            probs = action_probs ** (1.0 / temperature)
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                valid = board.get_valid_moves()
                idx = np.random.choice(len(valid))
                r, c = valid[idx]
                action = r * board.cols + c
                return (r, c), action_probs

            action = np.random.choice(len(probs), p=probs)

        row = action // board.cols
        col = action % board.cols
        return (row, col), action_probs
