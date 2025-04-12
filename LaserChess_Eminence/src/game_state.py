import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Define piece types as integers
class PieceType(Enum):
    EMPTY = 0
    KING = 1  # The piece that needs to be protected (no direction/reflection)
    LASER = 2  # Laser emitter
    MIRROR = 3  # Mirror Piece/Deflector - one 45° mirror face
    SWITCH = 4  # Switch - Mirror that can be either / or \
    OBELISK = 5  # Defender - No mirrors, blocks laser (no orientation)


# Define players as integers
class Player(Enum):
    NONE = 0
    PLAYER1 = 1  # Red player
    PLAYER2 = 2  # Blue player


# Define piece orientations
class Orientation(Enum):
    # For Switch
    FORWARD_SLASH = 0  # / switch
    BACK_SLASH = 1  # \ switch

    # For Laser
    UP = 0  # For laser facing up
    RIGHT = 1  # For laser facing right
    DOWN = 2  # For laser facing down
    LEFT = 3  # For laser facing left

    # For Mirror
    FORWARD_SLASH_LEFT = 0  # /M - Reflects UP to LEFT and LEFT to UP
    BACK_SLASH_RIGHT = 1  # M\ - Reflects UP to RIGHT and RIGHT to UP
    FORWARD_SLASH_RIGHT = 2  # M/ - Reflects DOWN to RIGHT and RIGHT to DOWN
    BACK_SLASH_LEFT = 3  # \M - Reflects DOWN to LEFT and LEFT to DOWN

    # For king and obelisk
    NONE = 0  # No orientation needed


# Define action types as integers
class ActionType(Enum):
    MOVE = 0  # Move a piece to a new position
    ROTATE_CW = 1  # Rotate a piece clockwise
    ROTATE_CCW = 2  # Rotate a piece counter-clockwise


@dataclass
class Action:
    """Represents a player action"""
    action_type: int  # 0=MOVE, 1=ROTATE_CW, 2=ROTATE_CCW
    piece_pos: Tuple[int, int]  # (row, col)
    target_pos: Optional[Tuple[int, int]] = None  # Only needed for MOVE action

    def __str__(self):
        action_names = ["Move", "Rotate CW", "Rotate CCW"]
        if self.action_type == 0:  # MOVE
            return f"{action_names[self.action_type]} {self.piece_pos} to {self.target_pos}"
        else:  # ROTATE
            return f"{action_names[self.action_type]} {self.piece_pos}"


class GameState:
    def __init__(self, rows=8, cols=10):
        self.rows = rows
        self.cols = cols

        # Board representation using numpy arrays for efficiency:
        # Each cell contains 3 values:
        # [piece_type, player, orientation]
        # where piece_type: 0=empty, 1=king, 2=laser, 3=mirror, 4=switch, 5=obelisk
        # player: 0=none, 1=player1 (red), 2=player2 (blue)
        # orientation:
        #   For laser: 0=up, 1=right, 2=down, 3=left
        #   For switch: 0=/, 1=\
        #   For mirror: 0=/M, 1=M\, 2=M/, 3=\M
        #   For king/obelisk: 0 (no orientation)
        self.board = np.zeros((rows, cols, 3), dtype=np.int8)

        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = 0  # 0=none, 1=player1, 2=player2

        # Initialize the board with pieces
        self._setup_board_custom()

    def _setup_board_custom(self):
        """
        Setup the initial board configuration with custom positions and orientations
        based on the provided image.

        piece_type: 0=empty, 1=king, 2=laser, 3=mirror/deflector, 4=switch, 5=obelisk/defender
        player: 0=none, 1=player1 (red), 2=player2 (blue)
        orientation:
          For laser: 0=up, 1=right, 2=down, 3=left
          For switch: 0=/, 1=\
          For mirror: 0=/M, 1=M\, 2=M/, 3=\M
          For king/obelisk: 0 (no orientation)
        """
        # Clear the board first
        self.board = np.zeros((self.rows, self.cols, 3), dtype=np.int8)

        # Player 1 (Red) pieces
        self.board[0, 0] = [2, 1, 2]  # Laser red at 0,0 facing down
        self.board[0, 4] = [5, 1, 0]  # Red Obelisk at 0,4 (no orientation)
        self.board[0, 5] = [1, 1, 0]  # Red King at 0,5 (no orientation)
        self.board[0, 6] = [5, 1, 0]  # Red Obelisk at 0,6 (no orientation)
        self.board[0, 7] = [3, 1, 2]  # Red Deflector at 0,7 (M/)
        self.board[1, 2] = [3, 1, 3]  # Red Deflector at 1,2 (\M)
        self.board[3, 0] = [3, 1, 1]  # Red Deflector at 3,0 (M\)
        self.board[3, 4] = [4, 1, 0]  # Red Switch at 3,4 (/)
        self.board[3, 5] = [4, 1, 1]  # Red Switch at 3,5 (\)
        self.board[3, 7] = [3, 1, 2]  # Red Deflector at 3,7 (M/)
        self.board[4, 0] = [3, 1, 2]  # Red Deflector at 4,0 (M/)
        self.board[4, 7] = [3, 1, 1]  # Red Deflector at 4,7 (M\)
        self.board[5, 6] = [3, 1, 2]  # Red Deflector at 5,6 (M/)

        # Player 2 (Blue) pieces
        self.board[7, 9] = [2, 2, 0]  # Laser blue at 7,9 facing up
        self.board[7, 3] = [5, 2, 0]  # Blue Obelisk at 7,3 (no orientation)
        self.board[7, 4] = [1, 2, 0]  # Blue King at 7,4 (no orientation)
        self.board[7, 5] = [5, 2, 0]  # Blue Obelisk at 7,5 (no orientation)
        self.board[7, 2] = [3, 2, 0]  # Blue Deflector at 7,2 (/M)
        self.board[6, 7] = [3, 2, 1]  # Blue Deflector at 6,7 (M\)
        self.board[2, 3] = [3, 2, 0]  # Blue Deflector at 2,3 (/M)
        self.board[3, 2] = [3, 2, 3]  # Blue Deflector at 3,2 (\M)
        self.board[3, 9] = [3, 2, 0]  # Blue Deflector at 3,9 (/M)
        self.board[4, 2] = [3, 2, 0]  # Blue Deflector at 4,2 (/M)
        self.board[4, 4] = [4, 2, 0]  # Blue Switch at 4,4 (/)
        self.board[4, 5] = [4, 2, 1]  # Blue Switch at 4,5 (\)
        self.board[4, 9] = [3, 2, 0]  # Blue Deflector at 4,9 (/M)

    def copy(self):
        """Create a deep copy of the current game state"""
        new_state = GameState(self.rows, self.cols)
        new_state.board = self.board.copy()  # Numpy array copy
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        return new_state

    def is_valid_position(self, row, col):
        """Check if a position is valid on the board"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_piece_at(self, row, col):
        """
        Get the piece information at the specified position
        Returns [piece_type, player, orientation]
        """
        if self.is_valid_position(row, col):
            return self.board[row, col]
        return None

    def is_empty(self, row, col):
        """Check if a position is empty"""
        return self.board[row, col, 0] == 0

    def apply_action(self, action):
        """Apply an action to the current state and return a new state"""
        # Create a copy of the current state
        new_state = self.copy()

        row, col = action.piece_pos
        piece = new_state.board[row, col]
        piece_type = piece[0]

        if piece[0] == 0 or piece[1] != self.current_player:
            # Invalid action: no piece or not player's piece
            return self  # Return unchanged state

        if action.action_type == 0:  # MOVE
            target_row, target_col = action.target_pos

            # Check if target position is valid and empty
            if (new_state.is_valid_position(target_row, target_col) and
                    new_state.is_empty(target_row, target_col)):

                # Check move distance (adjacent only, including diagonals)
                if abs(target_row - row) <= 1 and abs(target_col - col) <= 1:
                    # Move the piece
                    new_state.board[target_row, target_col] = piece
                    new_state.board[row, col] = [0, 0, 0]  # Empty the source cell
                else:
                    # Invalid move distance
                    return self
            else:
                # Invalid target position
                return self

        elif action.action_type == 1 or action.action_type == 2:  # ROTATE_CW or ROTATE_CCW
            # Only rotate pieces that have orientations (laser, mirror, switch)
            if piece_type == 1 or piece_type == 5:  # King or Obelisk
                # These pieces don't rotate
                return self

            if piece_type == 2:  # Laser
                # Laser has 4 orientations (0=up, 1=right, 2=down, 3=left)
                if action.action_type == 1:  # Rotate CW
                    new_state.board[row, col, 2] = (piece[2] + 1) % 4
                else:  # Rotate CCW
                    new_state.board[row, col, 2] = (piece[2] - 1) % 4
            elif piece_type == 3:  # Mirror
                # Mirrors have 4 orientations (0=/M, 1=M\, 2=M/, 3=\M)
                if action.action_type == 1:  # Rotate CW
                    new_state.board[row, col, 2] = (piece[2] + 1) % 4
                else:  # Rotate CCW
                    new_state.board[row, col, 2] = (piece[2] - 1) % 4
            elif piece_type == 4:  # Switch
                # Switches only have 2 orientations (0=/, 1=\)
                # Toggle between 0 and 1
                new_state.board[row, col, 2] = 1 - piece[2]

        # Fire laser and check for game over condition
        new_state._fire_laser()

        # If game is not over, switch player
        if not new_state.game_over:
            new_state.switch_player()

        return new_state

    def _is_valid_reflection(self, laser_dir, piece_type, mirror_orientation):
        """
        Determine if the laser hits the mirror/deflector on a reflective side

        Mirror orientations:
        - 0: /M (Forward slash on left) - Reflects UP to LEFT and LEFT to UP
        - 1: M\ (Back slash on right) - Reflects UP to RIGHT and RIGHT to UP
        - 2: M/ (Forward slash on right) - Reflects DOWN to RIGHT and RIGHT to DOWN
        - 3: \M (Back slash on left) - Reflects DOWN to LEFT and LEFT to DOWN

        Switch orientations:
        - 0: Forward slash (/)
        - 1: Back slash (\)

        Laser directions:
        - 0: UP
        - 1: RIGHT
        - 2: DOWN
        - 3: LEFT

        Returns True if the laser hits a reflective side, False otherwise
        """
        # Switch pieces reflect from all sides
        if piece_type == 4:  # SWITCH
            return True

        # Mirror pieces reflect only from specific sides
        if piece_type == 3:  # MIRROR
            # FIXED: These were the wrong direction combinations that could be reflected
            reflective_sides = {
                2: {0, 3},  # /M: UP or LEFT
                3: {0, 1},  # M\: UP or RIGHT
                0: {2, 1},  # M/: DOWN or RIGHT
                1: {2, 3},  # \M: DOWN or LEFT
            }
            return laser_dir in reflective_sides.get(mirror_orientation, set())

        # Default case
        return False

    def _fire_laser(self):
        """
        Fire the laser of the current player and determine if it hits anything.
        This updates the game state if an opponent's king is hit or a deflector
        is hit on its non-reflective side.
        """
        # Find the laser for the current player
        laser_pos = None
        laser_orientation = None

        for r in range(self.rows):
            for c in range(self.cols):
                if (self.board[r, c, 0] == 2 and  # Laser piece
                        self.board[r, c, 1] == self.current_player):  # Current player's piece
                    laser_pos = (r, c)
                    laser_orientation = self.board[r, c, 2]
                    break

        if laser_pos is None:
            return

        # Initial laser direction based on orientation
        laser_dir = laser_orientation  # Same numerical values

        # Start from the position in front of the laser
        curr_row, curr_col = laser_pos

        # Move to the first position in laser direction
        if laser_dir == 0:  # UP
            curr_row -= 1
        elif laser_dir == 1:  # RIGHT
            curr_col += 1
        elif laser_dir == 2:  # DOWN
            curr_row += 1
        else:  # LEFT
            curr_col -= 1

        # Follow the laser path until it hits something or goes out of bounds
        while self.is_valid_position(curr_row, curr_col):
            piece_type = self.board[curr_row, curr_col, 0]
            piece_player = self.board[curr_row, curr_col, 1]

            if piece_type == 0:  # Empty space
                # Continue in same direction
                if laser_dir == 0:  # UP
                    curr_row -= 1
                elif laser_dir == 1:  # RIGHT
                    curr_col += 1
                elif laser_dir == 2:  # DOWN
                    curr_row += 1
                else:  # LEFT (3)
                    curr_col -= 1
                continue

            # Hit a piece
            if piece_type == 1:  # KING
                # Game over if opponent's king is hit
                if piece_player != self.current_player:
                    self.game_over = True
                    self.winner = self.current_player
                return

            elif piece_type == 5:  # OBELISK
                # Obelisk blocks the laser
                return

            elif piece_type == 3 or piece_type == 4:  # MIRROR/DEFLECTOR or SWITCH
                # Get mirror/switch orientation
                piece_orientation = self.board[curr_row, curr_col, 2]
                # Check if the laser hits a reflective side
                if not self._is_valid_reflection(laser_dir, piece_type, piece_orientation):
                    # The deflector is hit on a non-reflective side and is destroyed
                    # Set this position to empty
                    self.board[curr_row, curr_col] = [0, 0, 0]
                    return

                # Calculate new laser direction based on reflection
                if piece_type == 3:  # MIRROR
                    laser_dir = self._reflect_from_mirror(laser_dir, piece_orientation)
                else:  # SWITCH
                    laser_dir = self._reflect_from_switch(laser_dir, piece_orientation)

            # Move to next position based on new direction
            if laser_dir == 0:  # UP
                curr_row -= 1
            elif laser_dir == 1:  # RIGHT
                curr_col += 1
            elif laser_dir == 2:  # DOWN
                curr_row += 1
            else:  # LEFT (3)
                curr_col -= 1

    def _reflect_from_mirror(self, laser_dir, mirror_orientation):
        """
        Determine new laser direction after hitting a mirror

        Mirror orientations:
        - 0: /M (Forward slash on left) - Reflects UP to LEFT and LEFT to UP
        - 1: M\ (Back slash on right) - Reflects UP to RIGHT and RIGHT to UP
        - 2: M/ (Forward slash on right) - Reflects DOWN to RIGHT and RIGHT to DOWN
        - 3: \M (Back slash on left) - Reflects DOWN to LEFT and LEFT to DOWN

        Laser directions:
        - 0: UP
        - 1: RIGHT
        - 2: DOWN
        - 3: LEFT
        """
        # FIXED: Corrected the reflection rules to match the mirror orientations
        if mirror_orientation == 0:
            if laser_dir == 1:
                return 0
            elif laser_dir == 2:
                return 3
        if mirror_orientation == 1:
            if laser_dir == 2:
                return 1
            elif laser_dir == 3:
                return 0
        if mirror_orientation == 2:
            if laser_dir == 3:
                return 2
            elif laser_dir == 0:
                return 1
        if mirror_orientation == 3:
            if laser_dir == 1:
                return 2
            elif laser_dir == 0:
                return 3

    def _reflect_from_switch(self, laser_dir, switch_orientation):
        """
        Determine new laser direction after hitting a switch

        Switch orientations:
        - 0: Forward slash (/)
        - 1: Back slash (\)

        Laser directions:
        - 0: UP
        - 1: RIGHT
        - 2: DOWN
        - 3: LEFT
        """
        if switch_orientation == 0:  # Forward slash (/)
            # UP→LEFT, RIGHT→DOWN, DOWN→RIGHT, LEFT→UP
            forward_slash_reflections = {0: 3, 1: 2, 2: 1, 3: 0}
            return forward_slash_reflections[laser_dir]
        else:  # Back slash (\)
            # UP→RIGHT, RIGHT→UP, DOWN→LEFT, LEFT→DOWN
            back_slash_reflections = {0: 1, 1: 0, 2: 3, 3: 2}
            return back_slash_reflections[laser_dir]

    def get_valid_actions(self):
        """Get all valid actions for the current player"""
        valid_actions = []

        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r, c]
                piece_type = piece[0]

                if piece_type == 0 or piece[1] != self.current_player:
                    continue  # Empty cell or not current player's piece

                # Only pieces with orientations can rotate
                if piece_type == 2 or piece_type == 3 or piece_type == 4:  # Laser, Mirror, or Switch
                    valid_actions.append(Action(1, (r, c)))  # Rotate CW
                    valid_actions.append(Action(2, (r, c)))  # Rotate CCW

                # All pieces except laser can move
                if piece_type != 2:  # Not a laser
                    # Check adjacent squares (including diagonals)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip current position

                            target_r, target_c = r + dr, c + dc

                            if (self.is_valid_position(target_r, target_c) and
                                    self.is_empty(target_r, target_c)):
                                valid_actions.append(Action(
                                    0,  # MOVE
                                    (r, c),
                                    (target_r, target_c)
                                ))

        return valid_actions

    def get_laser_path(self):
        """
        Calculate and return the path of the laser from the current player
        Returns list of (row, col) positions the laser travels through
        """
        # Find the laser for the current player
        laser_pos = None
        laser_orientation = None

        for r in range(self.rows):
            for c in range(self.cols):
                if (self.board[r, c, 0] == 2 and  # Laser piece
                        self.board[r, c, 1] == self.current_player):  # Current player's piece
                    laser_pos = (r, c)
                    laser_orientation = self.board[r, c, 2]
                    break

        if laser_pos is None:
            return []

        # Initial laser direction based on orientation
        laser_dir = laser_orientation  # Same numerical values

        # Start from the laser position
        curr_row, curr_col = laser_pos
        path = [(curr_row, curr_col)]  # Start with the laser position

        # Move to the first position in laser direction
        if laser_dir == 0:  # UP
            curr_row -= 1
        elif laser_dir == 1:  # RIGHT
            curr_col += 1
        elif laser_dir == 2:  # DOWN
            curr_row += 1
        else:  # LEFT
            curr_col -= 1

        # Follow the laser path until it hits something or goes out of bounds
        while self.is_valid_position(curr_row, curr_col):
            # Add position to path
            path.append((curr_row, curr_col))

            piece_type = self.board[curr_row, curr_col, 0]

            if piece_type == 0:  # Empty space
                # Continue in same direction
                if laser_dir == 0:  # UP
                    curr_row -= 1
                elif laser_dir == 1:  # RIGHT
                    curr_col += 1
                elif laser_dir == 2:  # DOWN
                    curr_row += 1
                else:  # LEFT (3)
                    curr_col -= 1
                continue

            # Hit a piece (end of path or reflection)
            if piece_type == 1 or piece_type == 5:  # KING or OBELISK
                # Path ends here
                return path

            elif piece_type == 3 or piece_type == 4:  # MIRROR/DEFLECTOR or SWITCH
                # Get mirror/switch orientation
                piece_orientation = self.board[curr_row, curr_col, 2]

                # Check if the laser hits a reflective side
                if not self._is_valid_reflection(laser_dir, piece_type, piece_orientation):
                    # The deflector is hit on a non-reflective side and is destroyed
                    # Path ends here
                    return path

                # Calculate new laser direction based on reflection
                if piece_type == 3:  # MIRROR
                    laser_dir = self._reflect_from_mirror(laser_dir, piece_orientation)
                else:  # SWITCH
                    laser_dir = self._reflect_from_switch(laser_dir, piece_orientation)

            # Move to next position based on new direction
            if laser_dir == 0:  # UP
                curr_row -= 1
            elif laser_dir == 1:  # RIGHT
                curr_col += 1
            elif laser_dir == 2:  # DOWN
                curr_row += 1
            else:  # LEFT (3)
                curr_col -= 1

        return path

    def switch_player(self):
        """Switch the current player"""
        self.current_player = 3 - self.current_player  # 1->2, 2->1

    def get_numerical_state(self):
        """Return a flattened numerical representation of the state for MCTS"""
        # Return the numpy array directly - already numerical
        # We add current player as the last element
        flat_board = self.board.flatten()
        return np.append(flat_board, self.current_player)

    def __str__(self):
        """String representation of the game state for debugging"""
        piece_names = ["Empty", "King", "Laser", "Mirror", "Switch", "Obelisk"]

        # Orientation representations
        def get_orientation_symbol(piece_type, orientation):
            if piece_type == 1 or piece_type == 5:  # King or Obelisk
                return " "  # No orientation
            elif piece_type == 2:  # Laser
                return ["↑", "→", "↓", "←"][orientation]
            elif piece_type == 3:  # Mirror
                return ["/M", "M\\", "M/", "\\M"][orientation]
            elif piece_type == 4:  # Switch
                return ["/", "\\"][orientation]
            return " "

        result = f"Current player: {self.current_player}\n"
        result += f"Game over: {self.game_over}, Winner: {self.winner}\n"

        # Column headers
        result += "  "
        for c in range(self.cols):
            result += f"{c} "
        result += "\n"

        for r in range(self.rows):
            result += f"{r} "  # Row header
            for c in range(self.cols):
                piece = self.board[r, c]
                if piece[0] == 0:  # Empty
                    result += ". "
                else:
                    # Get piece type, player, and orientation symbol
                    piece_type = piece[0]
                    player = piece[1]
                    orientation_symbol = get_orientation_symbol(piece_type, piece[2])

                    # Format as [type,player,orientation]
                    result += f"[{piece_names[piece_type][0]}{player}{orientation_symbol}] "
            result += "\n"
        return result