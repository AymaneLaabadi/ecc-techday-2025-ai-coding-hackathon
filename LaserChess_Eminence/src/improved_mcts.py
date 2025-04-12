import numpy as np
import time
import random
import hashlib
import math
import sys
from game_state import GameState, Action

# Increase recursion limit to handle deeper game trees
sys.setrecursionlimit(10000)


class LaserChessAdapter:
    def __init__(self, state):
        self.state = state

    def get_legal_actions(self):
        """Get legal actions from the current state"""
        # Only return actions that are truly valid
        return self.state.get_valid_actions()

    def is_terminal(self):
        """Check if this state is terminal (game over)"""
        return self.state.game_over

    def get_reward(self):
        """Get a more detailed reward signal based on game state"""
        if self.state.game_over:
            # Terminal state rewards (highest priority)
            if self.state.winner == 0:
                return 0  # Draw
            elif self.state.winner == self.state.current_player:
                return 1.0  # Win
            else:
                return -1.0  # Loss

        # Non-terminal rewards (intermediate progress rewards)
        reward = 0.0

        # Count pieces for each player
        player_pieces = {1: 0, 2: 0}
        player_deflectors = {1: 0, 2: 0}

        # Track kings for laser proximity calculation
        king_positions = {1: None, 2: None}
        laser_positions = {1: None, 2: None}
        laser_dirs = {1: None, 2: None}

        # Scan the board
        for r in range(self.state.rows):
            for c in range(self.state.cols):
                piece_type = self.state.board[r, c, 0]
                if piece_type == 0:  # Empty
                    continue

                player = self.state.board[r, c, 1]

                # Count pieces
                player_pieces[player] = player_pieces.get(player, 0) + 1

                # Count deflectors specifically
                if piece_type == 3:  # Mirror/deflector
                    player_deflectors[player] = player_deflectors.get(player, 0) + 1

                # Track king positions
                elif piece_type == 1:  # King
                    king_positions[player] = (r, c)

                # Track laser positions and directions
                elif piece_type == 2:  # Laser
                    laser_positions[player] = (r, c)
                    laser_dirs[player] = self.state.board[r, c, 2]  # Direction

        # Material advantage (minor reward)
        own_pieces = player_pieces.get(self.state.current_player, 0)
        opp_pieces = player_pieces.get(3 - self.state.current_player, 0)
        piece_advantage = own_pieces - opp_pieces
        reward += 0.01 * piece_advantage

        # Deflector advantage (more important than general pieces)
        own_deflectors = player_deflectors.get(self.state.current_player, 0)
        opp_deflectors = player_deflectors.get(3 - self.state.current_player, 0)
        deflector_advantage = own_deflectors - opp_deflectors
        reward += 0.02 * deflector_advantage

        # Laser-to-king proximity (most important intermediate reward)
        # This encourages positioning the laser to target the opponent's king
        own_player = self.state.current_player
        opp_player = 3 - own_player

        # If we have a laser and opponent has a king, calculate proximity
        if laser_positions[own_player] and king_positions[opp_player]:
            # Calculate if the opponent's king is in potential danger
            # (This is a simplified proxy - for a full implementation,
            # you'd want to actually trace the laser path)
            laser_r, laser_c = laser_positions[own_player]
            king_r, king_c = king_positions[opp_player]
            laser_dir = laser_dirs[own_player]

            # Basic proximity - closer is better
            distance = abs(laser_r - king_r) + abs(laser_c - king_c)
            proximity_reward = 0.05 * (10 - min(distance, 10)) / 10
            reward += proximity_reward

            # Direction alignment bonus - if laser points roughly toward king
            aligned = False
            if laser_dir == 0 and king_r < laser_r:  # UP
                aligned = True
            elif laser_dir == 1 and king_c > laser_c:  # RIGHT
                aligned = True
            elif laser_dir == 2 and king_r > laser_r:  # DOWN
                aligned = True
            elif laser_dir == 3 and king_c < laser_c:  # LEFT
                aligned = True

            if aligned:
                reward += 0.03

        # Self-protection reward - discouraging vulnerable king positions
        # Calculate if our king is in potential danger
        if laser_positions[opp_player] and king_positions[own_player]:
            laser_r, laser_c = laser_positions[opp_player]
            king_r, king_c = king_positions[own_player]

            # Basic proximity - further is better for our king
            distance = abs(laser_r - king_r) + abs(laser_c - king_c)
            protection_reward = 0.03 * min(distance, 10) / 10
            reward += protection_reward

        return reward

    def apply_action(self, action):
        """Apply an action with strict validation to prevent invalid moves"""
        if action is None:
            # Invalid action
            return LaserChessAdapter(self.state)

        # Apply the action to get a new state
        new_state = self.state.apply_action(action)

        # Check if the action is valid (changes player or ends game)
        if new_state.current_player == self.state.current_player and not new_state.game_over:
            # This is an invalid move that doesn't progress the game
            # Return the original state to signal invalidity
            return LaserChessAdapter(self.state)

        return LaserChessAdapter(new_state)

    def get_current_player(self):
        """Get the current player"""
        return self.state.current_player

    def get_state_hash(self):
        """Get a unique hash for the current state"""
        state_rep = self.state.get_numerical_state()
        state_str = np.array2string(state_rep, precision=0, separator=',')
        return hashlib.md5(state_str.encode()).hexdigest()


class PersistentMCTS:
    def __init__(self, exploration_weight=1.4, time_limit=1.0):
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self.root = None
        self.tree = {}
        self.invalid_moves = {}  # Track known invalid moves
        self.visit_threshold = 5  # Threshold for pruning low-visit nodes
        self.progressive_widening_base = 2  # Base for progressive widening
        self.progressive_widening_exponent = 0.5  # Exponent for progressive widening

    def initialize_with_state(self, state_adapter):
        """Initialize the MCTS tree with a starting state"""
        state_hash = state_adapter.get_state_hash()
        if state_hash not in self.tree:
            self.tree[state_hash] = {
                'N': 0,
                'Q': 0,
                'children': {},
                'actions': state_adapter.get_legal_actions(),
                'player': state_adapter.get_current_player(),
                'terminal': state_adapter.is_terminal(),
                'reward': state_adapter.get_reward() if state_adapter.is_terminal() else None,
                'state': state_adapter.state,
                'parent': None,  # Track parent for more efficient backpropagation
                'depth': 0,  # Track depth in tree for better exploration strategies
                'valid_visit_count': 0  # Count of valid simulations (useful for invalid move detection)
            }
        self.root = state_hash

    def update_root_for_state(self, state_adapter):
        """Update the root node to match the current game state"""
        state_hash = state_adapter.get_state_hash()
        print(f"Updating root to state hash: {state_hash}")

        if state_hash in self.tree:
            # We already know this state, just use it as root
            self.root = state_hash
            # Reset the depth of this node and all its children
            self._reset_depth(state_hash)
        else:
            # New state, initialize it
            self.initialize_with_state(state_adapter)

            # Periodically prune the tree to avoid excessive memory use
            if len(self.tree) > 10000:  # Arbitrary threshold
                self._prune_tree()

    def _reset_depth(self, node_hash, current_depth=0, visited=None):
        """Reset the depth of a node and all its children (avoid recursion issues)"""
        if visited is None:
            visited = set()

        if node_hash in visited or node_hash not in self.tree:
            return

        visited.add(node_hash)
        node = self.tree[node_hash]
        node['depth'] = current_depth

        # Process children iteratively rather than recursively
        for action_str in node['children']:
            child_hash = node['children'][action_str]
            if child_hash in self.tree and child_hash not in visited:
                child_node = self.tree[child_hash]
                child_node['parent'] = node_hash  # Ensure parent is correctly set
                # Queue this child for later processing instead of immediate recursion
                to_process = [(child_hash, current_depth + 1)]

                # Process the queue iteratively
                while to_process:
                    next_hash, next_depth = to_process.pop(0)
                    if next_hash in visited or next_hash not in self.tree:
                        continue

                    visited.add(next_hash)
                    next_node = self.tree[next_hash]
                    next_node['depth'] = next_depth

                    # Add this node's children to the queue
                    for next_action_str in next_node['children']:
                        next_child_hash = next_node['children'][next_action_str]
                        if next_child_hash in self.tree and next_child_hash not in visited:
                            next_child_node = self.tree[next_child_hash]
                            next_child_node['parent'] = next_hash
                            to_process.append((next_child_hash, next_depth + 1))

    def _prune_tree(self):
        """Prune nodes with low visit counts to manage memory usage"""
        keep_nodes = set([self.root])  # Always keep the root
        queue = [self.root]

        # First, identify nodes to keep using BFS to a certain depth
        max_depth_to_keep = 4  # Arbitrary depth limit

        while queue:
            node_hash = queue.pop(0)
            if node_hash not in self.tree:
                continue

            node = self.tree[node_hash]

            # Don't go deeper than max_depth from root
            if node.get('depth', 0) >= max_depth_to_keep:
                continue

            # Keep nodes with enough visits or that are terminal
            if node['N'] >= self.visit_threshold or node['terminal']:
                for action_str in node['children']:
                    child_hash = node['children'][action_str]
                    if child_hash not in keep_nodes:
                        keep_nodes.add(child_hash)
                        queue.append(child_hash)

        # Remove nodes not in the keep set
        pruned_count = 0
        for node_hash in list(self.tree.keys()):
            if node_hash not in keep_nodes:
                del self.tree[node_hash]
                pruned_count += 1

        print(f"Pruned {pruned_count} nodes from the tree, {len(self.tree)} remaining")

    def search(self, max_simulations=500):
        """Run the MCTS search algorithm to find the best action"""
        if self.root is None:
            raise ValueError("MCTS tree not initialized with a state")

        if self.root not in self.tree:
            print("Root node not found in tree")
            return None

        # Pre-filter invalid actions from the root node
        self._filter_invalid_actions(self.root)

        # Check if we have valid actions remaining
        node = self.tree[self.root]
        if not node['actions']:
            print(f"No valid actions at root node after filtering")
            return None

        # Now run the search
        end_time = time.time() + self.time_limit
        search_count = 0

        while time.time() < end_time and search_count < max_simulations:
            self._simulate()
            search_count += 1
            if search_count % 100 == 0:
                print(f"  Simulations: {search_count}")

        best_action = self._select_best_action(self.root, 0)
        print(f"MCTS completed {search_count} iterations")

        # Print top actions for debugging
        self._print_top_actions()

        return best_action

    def _filter_invalid_actions(self, node_hash):
        """Filter out invalid actions from a node"""
        if node_hash not in self.tree:
            return

        node = self.tree[node_hash]
        adapter = self._create_adapter_from_node(node_hash)

        if adapter.state is None:
            return

        # Create a fresh list of valid actions
        valid_actions = []

        for action in node['actions']:
            action_str = str(action)

            # Skip known invalid moves
            if action_str in self.invalid_moves.get(node_hash, {}):
                continue

            # Apply the action to check if it's valid
            child_adapter = adapter.apply_action(action)

            # A valid move either changes the player or ends the game
            if child_adapter.get_current_player() != adapter.get_current_player() or child_adapter.is_terminal():
                valid_actions.append(action)
            else:
                # Mark as invalid
                if node_hash not in self.invalid_moves:
                    self.invalid_moves[node_hash] = {}
                self.invalid_moves[node_hash][action_str] = True

        # Update the node with only valid actions
        node['actions'] = valid_actions

    def _print_top_actions(self, n=3):
        """Print the top N actions by visit count for debugging"""
        if self.root not in self.tree:
            return

        node = self.tree[self.root]
        actions = []

        for action in node['actions']:
            action_str = str(action)
            if action_str in node['children']:
                child_hash = node['children'][action_str]
                if child_hash in self.tree:
                    child = self.tree[child_hash]
                    expected_value = child['Q'] / max(child['N'], 1)
                    actions.append((action, child['N'], expected_value))

        # Sort by visit count
        actions.sort(key=lambda x: x[1], reverse=True)

        print("\nTop actions:")
        for i, (action, visits, value) in enumerate(actions[:n]):
            print(f"  {i + 1}. {action} - {visits} visits, value: {value:.3f}")
        print("")

    def _simulate(self):
        """Perform one simulation/playthrough"""
        # Use an iterative approach instead of recursion
        node_path, action_path = self._select(self.root)
        leaf_hash = node_path[-1]

        if leaf_hash not in self.tree:
            return

        if self.tree[leaf_hash]['terminal']:
            reward = self.tree[leaf_hash]['reward']
            self._backpropagate(node_path, action_path, reward)
            return

        child_hash, reward = self._expand(leaf_hash)

        if child_hash:
            node_path.append(child_hash)

        if not reward:
            reward = self._rollout(child_hash)

        self._backpropagate(node_path, action_path, reward)

    def _select(self, node_hash):
        """Select a path through the tree to a leaf node using UCB (iterative)"""
        node_path = [node_hash]
        action_path = []

        current_hash = node_hash
        depth = 0
        max_depth = 100  # Safety limit to prevent infinite loops

        while depth < max_depth:
            depth += 1

            if current_hash not in self.tree:
                break

            node = self.tree[current_hash]

            if node['terminal']:
                return node_path, action_path

            # First, make sure we only consider valid actions
            valid_actions = [a for a in node['actions']
                             if str(a) not in self.invalid_moves.get(current_hash, {})]

            if not valid_actions:
                return node_path, action_path

            # Progressive widening: limit actions considered based on visit count
            k = math.ceil(self.progressive_widening_base * (node['N'] ** self.progressive_widening_exponent))
            unexplored = [a for a in valid_actions if str(a) not in node['children']]

            # If we have unexplored actions and haven't exceeded the widening limit
            if unexplored and len(node['children']) < k:
                return node_path, action_path

            # Select best existing child using UCB
            best_action = self._select_best_action(current_hash, self.exploration_weight)

            # If all actions lead to invalid moves, return current path
            if best_action is None:
                return node_path, action_path

            action_path.append(best_action)
            best_action_str = str(best_action)

            # Check if this action is in children (it should be)
            if best_action_str not in node['children']:
                # This shouldn't happen, but just in case
                return node_path, action_path

            current_hash = node['children'][best_action_str]
            node_path.append(current_hash)

        return node_path, action_path

    def _select_best_action(self, node_hash, exploration_weight):
        """Select the best action from a node using UCB formula"""
        if node_hash not in self.tree:
            return None

        node = self.tree[node_hash]
        best_value = float('-inf')
        best_actions = []

        # Only consider actions not known to be invalid
        valid_actions = [a for a in node['actions']
                         if str(a) not in self.invalid_moves.get(node_hash, {})]

        if not valid_actions:
            return None

        # Apply UCB formula to each child
        for action in valid_actions:
            action_str = str(action)

            if action_str in node['children']:
                child_hash = node['children'][action_str]

                if child_hash in self.tree:
                    child_node = self.tree[child_hash]

                    if child_node['N'] > 0:
                        # UCB formula: Q/N + C * sqrt(ln(N_parent) / N_child)
                        exploitation = child_node['Q'] / child_node['N']

                        # Add exploration term if we're exploring
                        if exploration_weight > 0:
                            exploration = exploration_weight * math.sqrt(
                                math.log(max(node['N'], 1) + 1) / child_node['N'])
                            value = exploitation + exploration
                        else:
                            value = exploitation

                        # Negate value for min player (opponent)
                        if node['player'] != child_node['player']:
                            value = -value

                        # Track best action(s)
                        if value > best_value:
                            best_value = value
                            best_actions = [action]
                        elif value == best_value:
                            best_actions.append(action)

        # If we found valid best actions, return one randomly
        if best_actions:
            return random.choice(best_actions)

        # Otherwise, choose a random action from valid actions
        if valid_actions:
            return random.choice(valid_actions)

        # No valid actions found
        return None

    def _expand(self, node_hash):
        """Expand the tree by adding a new node"""
        if node_hash not in self.tree:
            return None, None

        node = self.tree[node_hash]

        # Get list of unexplored actions that aren't known to be invalid
        unexplored = [
            a for a in node['actions']
            if str(a) not in node['children'] and
               str(a) not in self.invalid_moves.get(node_hash, {})
        ]

        if not unexplored:
            return None, None

        # Choose a random unexplored action
        action = random.choice(unexplored)
        action_str = str(action)

        # Create a state adapter from the current node
        adapter = self._create_adapter_from_node(node_hash)
        if adapter.state is None:
            return None, None

        # Apply the chosen action to get the next state
        child_adapter = adapter.apply_action(action)

        # Check if the action is valid (changes the player or ends the game)
        if child_adapter.get_current_player() == adapter.get_current_player() and not child_adapter.is_terminal():
            # This is an invalid move, track it
            if node_hash not in self.invalid_moves:
                self.invalid_moves[node_hash] = {}
            self.invalid_moves[node_hash][action_str] = True
            return None, None

        # Valid move, get the new state's hash
        child_hash = child_adapter.get_state_hash()

        # Create a new node if we haven't seen this state before
        if child_hash not in self.tree:
            self.tree[child_hash] = {
                'N': 0,
                'Q': 0,
                'children': {},
                'actions': child_adapter.get_legal_actions(),
                'player': child_adapter.get_current_player(),
                'terminal': child_adapter.is_terminal(),
                'reward': child_adapter.get_reward() if child_adapter.is_terminal() else None,
                'state': child_adapter.state,
                'parent': node_hash,
                'depth': node.get('depth', 0) + 1,
                'valid_visit_count': 0
            }

        # Link the parent to this child
        node['children'][action_str] = child_hash

        # If the state is terminal, return its reward immediately
        reward = None
        if self.tree[child_hash]['terminal']:
            reward = self.tree[child_hash]['reward']

        return child_hash, reward

    def _rollout(self, node_hash):
        """Improved rollout with strategic move selection"""
        if node_hash is None or node_hash not in self.tree:
            return 0

        adapter = self._create_adapter_from_node(node_hash)
        if adapter.state is None:
            return 0

        depth = 0
        max_depth = 50
        accumulated_reward = 0
        discount_factor = 0.95  # For temporal discounting

        while not adapter.is_terminal() and depth < max_depth:
            legal_actions = adapter.get_legal_actions()
            if not legal_actions:
                break

            # Strategic move selection instead of pure random
            # For simplicity, sometimes choose randomly, sometimes choose "smartly"
            if random.random() < 0.7:  # 70% strategic, 30% random exploration
                # Apply each action and pick the one with highest immediate reward
                best_action = None
                best_reward = float('-inf')

                # Sample a subset of actions to consider (for efficiency)
                sample_size = min(5, len(legal_actions))
                action_sample = random.sample(legal_actions, sample_size)

                for action in action_sample:
                    next_adapter = adapter.apply_action(action)
                    if next_adapter.get_current_player() == adapter.get_current_player():
                        continue  # Skip invalid moves

                    # Get intermediate reward
                    action_reward = next_adapter.get_reward()
                    if action_reward > best_reward:
                        best_reward = action_reward
                        best_action = action

                # If we found a good action, use it
                if best_action:
                    action = best_action
                else:
                    action = random.choice(legal_actions)
            else:
                # Pure random exploration
                action = random.choice(legal_actions)

            # Apply the selected action
            new_adapter = adapter.apply_action(action)

            # Skip invalid moves
            if new_adapter.get_current_player() == adapter.get_current_player() and not new_adapter.is_terminal():
                continue

            # Get intermediate reward and accumulate with discount
            step_reward = new_adapter.get_reward()
            accumulated_reward += (discount_factor ** depth) * step_reward

            adapter = new_adapter
            depth += 1

        # Terminal state reward (weighted higher than intermediate rewards)
        if adapter.is_terminal():
            terminal_reward = adapter.get_reward() * 2  # Weight terminal states more heavily
            accumulated_reward += terminal_reward

        # Return the accumulated discounted reward
        return accumulated_reward

    def _backpropagate(self, node_path, action_path, reward):
        """Handle backpropagation with the improved reward system"""
        for i, node_hash in enumerate(node_path):
            if node_hash not in self.tree:
                continue

            node = self.tree[node_hash]
            node['N'] += 1
            node['valid_visit_count'] += 1

            # If this is a child node, adjust reward based on player perspective
            if i > 0:
                parent_hash = node_path[i - 1]
                if parent_hash in self.tree:
                    parent = self.tree[parent_hash]
                    if parent['player'] != node['player']:
                        # For opponent nodes, we need to negate rewards
                        # but preserve small intermediate rewards that benefit both
                        if abs(reward) > 0.1:  # Major rewards (wins/losses)
                            effective_reward = -reward
                        else:  # Minor intermediate rewards
                            # Negate but scale down for smoother learning
                            effective_reward = -reward * 0.8
                    else:
                        effective_reward = reward
                else:
                    effective_reward = reward
            else:
                effective_reward = reward

            # Update the quality value
            node['Q'] += effective_reward

    def penalize_invalid_move(self, action):
        """Mark an action as invalid to avoid selecting it again"""
        if self.root is None or action is None:
            return

        action_str = str(action)

        # Add this move to the invalid moves set
        if self.root not in self.invalid_moves:
            self.invalid_moves[self.root] = {}
        self.invalid_moves[self.root][action_str] = True

        # Also remove it from children if it exists
        if self.root in self.tree:
            node = self.tree[self.root]
            if action_str in node['children']:
                child_hash = node['children'][action_str]
                # Don't delete the child node - it might be referenced elsewhere
                # Just remove the connection
                del node['children'][action_str]

    def _create_adapter_from_node(self, node_hash):
        """Create a game state adapter from a node in the tree"""
        node = self.tree.get(node_hash)
        if node and 'state' in node:
            return LaserChessAdapter(node['state'])
        else:
            print(f"Warning: State not found for node {node_hash}")
            return LaserChessAdapter(None)

    def update_after_move(self, action):
        """Update the tree after a move has been made"""
        if self.root is None:
            print("Warning: Tree root is None, cannot update after move")
            return

        # Find the child state that matches this action
        action_str = str(action)
        if self.root in self.tree:
            node = self.tree[self.root]

            # If this action is already in the children, just update the root
            if action_str in node['children']:
                # Set the new root to the child state
                self.root = node['children'][action_str]
                # Reset depth starting from new root
                self._reset_depth(self.root)
                print(f"Tree updated for action: {action}")
            else:
                # Action not in tree, probably because it was newly explored
                # Create adapter and simulate the action
                adapter = self._create_adapter_from_node(self.root)
                if adapter.state is None:
                    print("Warning: Could not create adapter from root")
                    self.root = None
                    return

                # Apply the action to get the next state
                child_adapter = adapter.apply_action(action)

                # Get the new state's hash
                child_hash = child_adapter.get_state_hash()

                # Create a new node if we haven't seen this state before
                if child_hash not in self.tree:
                    self.tree[child_hash] = {
                        'N': 1,  # Initialize with single visit
                        'Q': 0,
                        'children': {},
                        'actions': child_adapter.get_legal_actions(),
                        'player': child_adapter.get_current_player(),
                        'terminal': child_adapter.is_terminal(),
                        'reward': child_adapter.get_reward() if child_adapter.is_terminal() else None,
                        'state': child_adapter.state,
                        'parent': self.root,
                        'depth': 0,  # Will be updated by reset_depth
                        'valid_visit_count': 1
                    }

                # Connect the parent to this child
                node['children'][action_str] = child_hash

                # Set the new root
                self.root = child_hash

                # Reset depth starting from new root
                self._reset_depth(self.root)
                print(f"Tree updated for new action: {action}")
        else:
            # If root not in tree (shouldn't happen), initialize a new tree
            print("Warning: Root not in tree, reinitializing")
            self.root = None