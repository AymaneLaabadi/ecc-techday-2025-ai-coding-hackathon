import os
import pickle
import time
import random
from improved_mcts import PersistentMCTS, LaserChessAdapter
from game_state import GameState, Action

# Global MCTS object to persist between moves and games
global_mcts = None

# Directory for storing AI knowledge
AI_KNOWLEDGE_DIR = "ai_knowledge"
DEFAULT_TREE_FILE = os.path.join(AI_KNOWLEDGE_DIR, "mcts_tree.pkl")
BACKUP_TREE_FILE = os.path.join(AI_KNOWLEDGE_DIR, "mcts_tree_backup.pkl")

# Track learning metrics
learning_stats = {
    'games_played': 0,
    'moves_made': 0,
    'wins': 0,
    'losses': 0,
    'last_save_time': 0
}


def ensure_knowledge_dir():
    if not os.path.exists(AI_KNOWLEDGE_DIR):
        os.makedirs(AI_KNOWLEDGE_DIR)


def save_tree(filename=DEFAULT_TREE_FILE, backup=True):
    """Save the MCTS tree to a file with optional backup"""
    global global_mcts, learning_stats

    if global_mcts is None:
        print("No tree to save")
        return False

    ensure_knowledge_dir()

    # Create a backup first if requested
    if backup and os.path.exists(filename):
        try:
            os.replace(filename, BACKUP_TREE_FILE)
            print(f"Created backup of previous tree")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")

    try:
        # Update stats before saving
        learning_stats['last_save_time'] = time.time()

        # Save both the tree and learning stats
        data_to_save = {
            'tree': global_mcts,
            'stats': learning_stats
        }

        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"MCTS tree saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving tree: {e}")
        # Try to restore from backup if we failed
        if backup and os.path.exists(BACKUP_TREE_FILE):
            try:
                os.replace(BACKUP_TREE_FILE, filename)
                print("Restored from backup after save failure")
            except:
                pass
        return False


def load_tree(filename=DEFAULT_TREE_FILE):
    """Load the MCTS tree from a file"""
    global global_mcts, learning_stats

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Handle both new and old format saves
        if isinstance(data, dict) and 'tree' in data:
            global_mcts = data['tree']
            learning_stats = data.get('stats', learning_stats)
        else:
            # Legacy format - just the tree
            global_mcts = data

        print(f"MCTS tree loaded from {filename}")
        print(f"AI has played {learning_stats['games_played']} games")

        # Add opening book moves to the loaded tree
        if global_mcts:
            add_book_moves()

        return True
    except Exception as e:
        print(f"Error loading tree: {e}")

        # Try loading from backup if main file failed
        if filename == DEFAULT_TREE_FILE and os.path.exists(BACKUP_TREE_FILE):
            print("Attempting to load from backup file...")
            try:
                with open(BACKUP_TREE_FILE, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, dict) and 'tree' in data:
                    global_mcts = data['tree']
                    learning_stats = data.get('stats', learning_stats)
                else:
                    global_mcts = data

                print(f"Successfully loaded from backup file")
                # Save to the main file to repair it
                save_tree(filename, backup=False)

                # Add opening book moves
                if global_mcts:
                    add_book_moves()

                return True
            except Exception as e:
                print(f"Failed to load from backup: {e}")

        global_mcts = None
        return False


def add_book_moves():
    """Add predefined strong opening moves to guide initial learning"""
    global global_mcts

    if global_mcts is None:
        return

    # Define some standard strong opening positions/moves for laser chess
    opening_book = {
        # Starting position hash (this is an example - you'll need the actual hash)
        "d39646354a4aa6cf898466cea1466071": [
            # Player 1 moves (red)
            Action(1, (3, 4)),  # Rotate switch at 3,4 clockwise
            Action(0, (0, 7), (1, 6)),  # Move mirror from 0,7 to 1,6
            Action(1, (3, 0)),  # Rotate deflector at 3,0 clockwise

            # Defensive moves
            Action(0, (0, 6), (1, 7)),  # Move obelisk to protect king
            Action(0, (3, 7), (2, 6)),  # Move deflector to better position
        ]
    }

    moves_added = 0

    # Add these positions to the MCTS tree with high initial values
    for state_hash, moves in opening_book.items():
        if state_hash in global_mcts.tree:
            node = global_mcts.tree[state_hash]
            for move in moves:
                move_str = str(move)

                # Skip if this book move is already in the tree
                if move_str in node['children']:
                    continue

                # Create a simulation of applying this move
                adapter = global_mcts._create_adapter_from_node(state_hash)
                if adapter.state is None:
                    continue

                # Apply the move and check if it's valid
                child_adapter = adapter.apply_action(move)
                if child_adapter.get_current_player() == adapter.get_current_player() and not child_adapter.is_terminal():
                    # Invalid move in our book, skip it
                    continue

                child_hash = child_adapter.get_state_hash()

                # Add to tree with high initial values to encourage selection
                if child_hash not in global_mcts.tree:
                    global_mcts.tree[child_hash] = {
                        'N': 50,  # High visit count to encourage selection
                        'Q': 40,  # Positive value to indicate a good move
                        'children': {},
                        'actions': child_adapter.get_legal_actions(),
                        'player': child_adapter.get_current_player(),
                        'terminal': child_adapter.is_terminal(),
                        'reward': child_adapter.get_reward() if child_adapter.is_terminal() else None,
                        'state': child_adapter.state,
                        'parent': state_hash,
                        'depth': 1,
                        'valid_visit_count': 50
                    }

                # Connect parent to child
                node['children'][move_str] = child_hash
                moves_added += 1

    if moves_added > 0:
        print(f"Added {moves_added} book moves to the MCTS tree")


def update_learning_stats(winner=None):
    """Update learning statistics after a game"""
    global learning_stats

    learning_stats['games_played'] += 1

    if winner == 1:  # AI is typically player 2
        learning_stats['losses'] += 1
    elif winner == 2:
        learning_stats['wins'] += 1


def ai_player_learning(state, time_limit=1.0, max_simulations=1000):
    """Enhanced AI player function with faster first move performance"""
    global global_mcts, learning_stats

    # Detect if this is the first move (don't rely on loading/saving that could block)
    is_first_move = False
    if state.current_player == 2:  # If AI is playing
        # Count pieces to check if we're at or near the starting position
        piece_count = 0
        for r in range(state.rows):
            for c in range(state.cols):
                if state.board[r, c, 0] > 0:
                    piece_count += 1
        # In starting position, we should have close to the full set of pieces
        is_first_move = (piece_count >= 26)

    # Use optimized parameters for the first move to make it almost instant
    if is_first_move:
        print("First AI move detected - using fast parameters")

        # Define some guaranteed good first moves
        fast_first_moves = [
            Action(1, (3, 2)),  # Rotate deflector at 3,2 clockwise
            Action(1, (4, 4)),  # Rotate switch at 4,4 clockwise
            Action(0, (7, 3), (6, 3)),  # Move obelisk forward to protect
            Action(0, (7, 5), (6, 5)),  # Move obelisk forward to protect
            Action(0, (7, 2), (6, 2)),  # Move deflector forward
        ]

        # Try each move until we find a valid one
        for move in fast_first_moves:
            new_state = state.apply_action(move)
            # Check if move is valid (changes player or ends game)
            if new_state.current_player != state.current_player or new_state.game_over:
                print(f"Using predefined first move: {move}")

                # Update the MCTS tree for this move (if it exists)
                if global_mcts and hasattr(global_mcts, 'update_after_move'):
                    try:
                        global_mcts.update_after_move(move)
                    except Exception as e:
                        print(f"Error updating tree (non-critical): {e}")

                # Update stats
                learning_stats['moves_made'] += 1
                return move

        # If we're here, none of our predefined moves worked
        # Use extremely limited MCTS for speed
        max_simulations = min(max_simulations, 100)
        time_limit = min(time_limit, 0.3)

    # Use regular parameters for non-first moves
    adapter = LaserChessAdapter(state)

    # Ensure MCTS tree is loaded/initialized
    if global_mcts is None:
        if not load_tree():
            # Create new tree with current parameters
            global_mcts = PersistentMCTS(exploration_weight=1.4, time_limit=time_limit)
            global_mcts.initialize_with_state(adapter)
    else:
        # Update root for current state
        global_mcts.exploration_weight = 1.4
        global_mcts.time_limit = time_limit
        try:
            global_mcts.update_root_for_state(adapter)
        except Exception as e:
            print(f"Error updating root: {e}")
            # If update fails, create a new tree
            global_mcts = PersistentMCTS(exploration_weight=1.4, time_limit=time_limit)
            global_mcts.initialize_with_state(adapter)

    # Get valid actions from the game state directly
    valid_actions = state.get_valid_actions()
    if not valid_actions:
        print("[ERROR] No valid actions available from the game state")
        return None

    # Run MCTS search with appropriate parameters
    action = None
    try:
        action = global_mcts.search(max_simulations=max_simulations)
    except Exception as e:
        print(f"Error in MCTS search: {e}")

    # If search failed, use random action
    if action is None:
        print("[WARNING] MCTS returned None, selecting random valid action")
        action = random.choice(valid_actions)

    # Validate the selected action
    new_state = state.apply_action(action)
    if new_state.current_player != state.current_player and not new_state.game_over:
        print(f"[Warning] AI chose invalid move: {action}. Using random fallback.")
        # Try a random move instead
        fallback_actions = [a for a in valid_actions if a != action]
        if fallback_actions:
            action = random.choice(fallback_actions)
            new_state = state.apply_action(action)

    # Update MCTS tree for the chosen action
    if hasattr(global_mcts, 'update_after_move'):
        try:
            global_mcts.update_after_move(action)
        except Exception as e:
            print(f"Error updating tree after move: {e}")

    # Update stats
    learning_stats['moves_made'] += 1

    # Don't auto-save during gameplay - this can cause freezing
    # We'll save only at game end in main.py

    return action


def reset_ai(clear_files=False):
    """Reset the AI's learning state"""
    global global_mcts, learning_stats
    global_mcts = None

    # Reset stats
    learning_stats = {
        'games_played': 0,
        'moves_made': 0,
        'wins': 0,
        'losses': 0,
        'last_save_time': 0
    }

    if clear_files:
        for file in [DEFAULT_TREE_FILE, BACKUP_TREE_FILE]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted {file}")
                except:
                    print(f"Failed to delete {file}")
    print("AI search tree has been reset")


# Compatibility alias
def ai_player(state, time_limit=1.0):
    return ai_player_learning(state, time_limit)


def get_learning_stats():
    """Get statistics about AI learning progress"""
    global learning_stats
    return dict(learning_stats)  # Return a copy