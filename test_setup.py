#!/usr/bin/env python3
"""Quick test to verify the training setup"""

import sys
import torch
from game.board import Board
from models.agentRL import AgentRL
from models.agentMiniMax import AgentMiniMax

def test_imports():
    print("✓ All imports successful")

def test_board():
    board = Board(rows=15, cols=15)
    print(f"✓ Board created: {board.rows}x{board.cols}")

def test_agents():
    board = Board(rows=15, cols=15)
    
    rl_agent = AgentRL(board)
    print(f"✓ RL Agent created")
    print(f"  - Device: {rl_agent.device}")
    print(f"  - Network parameters: {sum(p.numel() for p in rl_agent.network.parameters()):,}")
    
    minimax_agent = AgentMiniMax(board)
    print(f"✓ MiniMax Agent created")

def test_forward_pass():
    board = Board(rows=15, cols=15)
    rl_agent = AgentRL(board)
    
    # Test board to tensor conversion
    state = rl_agent.board_to_tensor(board)
    print(f"✓ Board to tensor: shape {state.shape}")
    
    # Test network forward pass
    with torch.no_grad():
        output = rl_agent.network(state.unsqueeze(0))
    print(f"✓ Network forward pass: output shape {output.shape}")

def test_move_selection():
    board = Board(rows=15, cols=15)
    rl_agent = AgentRL(board)
    
    # Test move selection
    move = rl_agent.select_move(player=1, use_epsilon=True)
    print(f"✓ Move selection: {move}")
    
    # Test valid moves
    valid_moves = rl_agent.get_valid_moves()
    print(f"✓ Valid moves count: {len(valid_moves)}")

def main():
    print("=" * 60)
    print("Testing Gomoku AI Training Setup")
    print("=" * 60)
    
    try:
        test_imports()
        test_board()
        test_agents()
        test_forward_pass()
        test_move_selection()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Ready to train.")
        print("=" * 60)
        print("\nTo start training, run:")
        print("  python -m pipeline.train")
        print("\nTo evaluate a trained model, run:")
        print("  python -m pipeline.inference")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
