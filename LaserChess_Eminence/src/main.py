import pygame
import pygame.gfxdraw
from ai_player_learning import save_tree
import sys
import math
from game_state import GameState, Action, PieceType, Player, Orientation, ActionType
from ai_player_learning import ai_player_learning

import threading
import time
from ai_player_learning import global_mcts, load_tree, PersistentMCTS, LaserChessAdapter


# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
BOARD_MARGIN = 50
CELL_SIZE = 75
PIECE_RADIUS = 30
INFO_HEIGHT = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (120, 120, 120)
LIGHT_GRAY = (230, 230, 230)
RED = (220, 60, 60)
BLUE = (60, 60, 220)
YELLOW = (220, 220, 60)
GREEN = (60, 220, 60)
LASER_RED = (255, 100, 100)
LASER_BLUE = (100, 100, 255)

# Player colors
PLAYER_COLORS = {
    1: RED,  # Player 1 - red
    2: BLUE  # Player 2 - blue
}

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Laser Chess")

# Create fonts
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 36, bold=True)


def draw_board(state):
    """Draw the chess board"""
    # Calculate board dimensions
    board_width = state.cols * CELL_SIZE
    board_height = state.rows * CELL_SIZE
    board_x = (SCREEN_WIDTH - board_width) // 2
    board_y = BOARD_MARGIN

    # Draw board background
    for r in range(state.rows):
        for c in range(state.cols):
            rect = pygame.Rect(
                board_x + c * CELL_SIZE,
                board_y + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            color = LIGHT_GRAY if (r + c) % 2 == 0 else GRAY
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, DARK_GRAY, rect, 1)  # Border

            # Draw coordinates
            coord_text = small_font.render(f"{r},{c}", True, DARK_GRAY)
            screen.blit(coord_text, (
                board_x + c * CELL_SIZE + 5,
                board_y + r * CELL_SIZE + CELL_SIZE - 20
            ))

    # Draw border around the board
    border_rect = pygame.Rect(
        board_x - 2,
        board_y - 2,
        board_width + 4,
        board_height + 4
    )
    pygame.draw.rect(screen, BLACK, border_rect, 2)

    return board_x, board_y


def draw_pieces(state, board_x, board_y):
    """Draw the chess pieces"""
    for r in range(state.rows):
        for c in range(state.cols):
            piece_type = state.board[r, c, 0]
            player = state.board[r, c, 1]
            orientation = state.board[r, c, 2]

            if piece_type == 0:  # Empty
                continue

            # Calculate piece center position
            center_x = board_x + c * CELL_SIZE + CELL_SIZE // 2
            center_y = board_y + r * CELL_SIZE + CELL_SIZE // 2

            # Draw based on piece type
            if piece_type == 1:  # King
                draw_king(center_x, center_y, player)
            elif piece_type == 2:  # Laser
                draw_laser(center_x, center_y, player, orientation)
            elif piece_type == 3:  # Mirror
                draw_mirror(center_x, center_y, player, orientation)
            elif piece_type == 4:  # Switch
                draw_switch(center_x, center_y, player, orientation)
            elif piece_type == 5:  # Obelisk
                draw_obelisk(center_x, center_y, player)


def draw_king(x, y, player):
    """Draw a king piece"""
    color = PLAYER_COLORS[player]

    # Draw circle
    pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
    pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)

    # Draw crown
    points = [
        (x - 14, y - 5),
        (x - 5, y - 15),
        (x, y - 3),
        (x + 5, y - 15),
        (x + 14, y - 5)
    ]
    pygame.draw.lines(screen, YELLOW, False, points, 6)

    # Draw "K" text
    text = font.render("K", True, YELLOW)
    text_rect = text.get_rect(center=(x, y + 4))
    screen.blit(text, text_rect)


def draw_laser(x, y, player, orientation):
    """Draw a laser piece"""
    color = PLAYER_COLORS[player]

    # Draw circle
    pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
    pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)

    # Draw direction indicator
    directions = [
        (0, -1),  # Up
        (1, 0),  # Right
        (0, 1),  # Down
        (-1, 0)  # Left
    ]

    dx, dy = directions[orientation]
    pygame.draw.line(
        screen,
        BLACK,
        (x, y),
        (x + dx * 18, y + dy * 40),
        20
    )

    # Draw arrowhead
    if orientation == 0:  # Up
        pygame.draw.polygon(screen, YELLOW, [
            (x, y - 25),
            (x - 5, y - 20),
            (x + 5, y - 20)
        ])
    elif orientation == 1:  # Right
        pygame.draw.polygon(screen, YELLOW, [
            (x + 25, y),
            (x + 20, y - 5),
            (x + 20, y + 5)
        ])
    elif orientation == 2:  # Down
        pygame.draw.polygon(screen, YELLOW, [
            (x, y + 25),
            (x - 5, y + 20),
            (x + 5, y + 20)
        ])
    elif orientation == 3:  # Left
        pygame.draw.polygon(screen, YELLOW, [
            (x - 25, y),
            (x - 20, y - 5),
            (x - 20, y + 5)
        ])

    # Draw "L" text
    text = font.render("L", True, WHITE)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)


def draw_mirror(x, y, player, orientation):
    """Draw a mirror piece"""
    color = PLAYER_COLORS[player]

    # Draw circle
    pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
    pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)

    # Draw mirror based on orientation with triangles
    # The hypotenuse represents the reflective surface
    # Orientation in GameState:
    # - 0: /M (Forward slash on left) - Reflects UP to LEFT and LEFT to UP
    # - 1: M\ (Back slash on right) - Reflects UP to RIGHT and RIGHT to UP
    # - 2: M/ (Forward slash on right) - Reflects DOWN to RIGHT and RIGHT to DOWN
    # - 3: \M (Back slash on left) - Reflects DOWN to LEFT and LEFT to DOWN
    if orientation == 2:  # /M
        pygame.draw.polygon(screen, WHITE, [
            (x - 15, y + 15),  # Bottom left
            (x + 15, y - 15),  # Top right
            (x - 15, y - 15)  # Top left
        ])
    elif orientation == 3:  # M\
        pygame.draw.polygon(screen, WHITE, [
            (x - 15, y - 15),  # Top left
            (x + 15, y - 15),  # Top right
            (x + 15, y + 15)  # Bottom right
        ])
    elif orientation == 0:  # M/
        pygame.draw.polygon(screen, WHITE, [
            (x + 15, y + 15),  # Bottom right
            (x + 15, y - 15),  # Top right
            (x - 15, y + 15)  # Bottom left
        ])
    elif orientation == 1:  # \M
        pygame.draw.polygon(screen, WHITE, [
            (x - 15, y - 15),  # Top left
            (x - 15, y + 15),  # Bottom left
            (x + 15, y + 15)  # Bottom right
        ])

    # Draw border for the triangle
    if orientation == 2:  # /M
        pygame.draw.line(screen, BLACK, (x - 15, y + 15), (x + 15, y - 15), 2)  # Hypotenuse
        pygame.draw.line(screen, BLACK, (x - 15, y - 15), (x + 15, y - 15), 1)  # Top
        pygame.draw.line(screen, BLACK, (x - 15, y - 15), (x - 15, y + 15), 1)  # Left
    elif orientation == 3:  # M\
        pygame.draw.line(screen, BLACK, (x - 15, y - 15), (x + 15, y + 15), 2)  # Hypotenuse
        pygame.draw.line(screen, BLACK, (x - 15, y - 15), (x + 15, y - 15), 1)  # Top
        pygame.draw.line(screen, BLACK, (x + 15, y - 15), (x + 15, y + 15), 1)  # Right
    elif orientation == 0:  # M/
        pygame.draw.line(screen, BLACK, (x - 15, y + 15), (x + 15, y - 15), 2)  # Hypotenuse
        pygame.draw.line(screen, BLACK, (x - 15, y + 15), (x + 15, y + 15), 1)  # Bottom
        pygame.draw.line(screen, BLACK, (x + 15, y - 15), (x + 15, y + 15), 1)  # Right
    elif orientation == 1:  # \M
        pygame.draw.line(screen, BLACK, (x - 15, y - 15), (x + 15, y + 15), 2)  # Hypotenuse
        pygame.draw.line(screen, BLACK, (x - 15, y - 15), (x - 15, y + 15), 1)  # Left
        pygame.draw.line(screen, BLACK, (x - 15, y + 15), (x + 15, y + 15), 1)  # Bottom

    # Draw "M" text
    text = font.render("M", True, BLACK)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)


def draw_switch(x, y, player, orientation):
    """Draw a switch piece"""
    color = PLAYER_COLORS[player]

    # Draw circle
    pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
    pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)

    # Draw switch based on orientation with triangles
    # Similar to mirrors but with a distinct color to show it's a switch
    if orientation == 0:  # S/
        pygame.draw.line(screen, WHITE, (x - 15, y - 15), (x + 15, y + 15), 8)
    elif orientation == 1:  # S\
        pygame.draw.line(screen, WHITE, (x - 15, y + 15), (x + 15, y - 15), 8)  #

    # Draw "S" text
    text = font.render("S", True, BLACK)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)


def draw_obelisk(x, y, player):
    """Draw an obelisk piece"""
    color = PLAYER_COLORS[player]

    # Draw circle
    pygame.draw.circle(screen, color, (x, y), PIECE_RADIUS)
    pygame.draw.circle(screen, BLACK, (x, y), PIECE_RADIUS, 2)

    # Draw obelisk shape
    pygame.draw.rect(screen, DARK_GRAY,
                     (x - 15, y - 15, 30, 30), 0)

    # Draw "O" text
    text = font.render("O", True, WHITE)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)


def draw_laser_path(state, board_x, board_y):
    """Draw the laser path on the board"""
    path = state.get_laser_path()
    if not path or len(path) <= 1:
        return

    # Determine laser color based on current player
    laser_color = LASER_RED if state.current_player == 1 else LASER_BLUE

    # Draw laser beam segments
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        start_x = board_x + c1 * CELL_SIZE + CELL_SIZE // 2
        start_y = board_y + r1 * CELL_SIZE + CELL_SIZE // 2
        end_x = board_x + c2 * CELL_SIZE + CELL_SIZE // 2
        end_y = board_y + r2 * CELL_SIZE + CELL_SIZE // 2

        # Draw laser line
        pygame.draw.line(screen, laser_color, (start_x, start_y), (end_x, end_y), 3)

        # Draw a small dot at each point of the path
        if i > 0:  # Skip the laser piece itself
            pygame.draw.circle(screen, laser_color, (start_x, start_y), 4)

    # Draw the end point
    last_r, last_c = path[-1]
    last_x = board_x + last_c * CELL_SIZE + CELL_SIZE // 2
    last_y = board_y + last_r * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, laser_color, (last_x, last_y), 4)


def draw_selected_highlight(row, col, board_x, board_y):
    """Highlight the selected piece"""
    rect = pygame.Rect(
        board_x + col * CELL_SIZE + 2,
        board_y + row * CELL_SIZE + 2,
        CELL_SIZE - 4,
        CELL_SIZE - 4
    )
    pygame.draw.rect(screen, YELLOW, rect, 3)


def draw_valid_moves(state, selected_pos, board_x, board_y):
    """Highlight valid move positions for the selected piece"""
    if not selected_pos:
        return

    row, col = selected_pos
    piece_type = state.board[row, col, 0]

    # Skip if it's a laser (can't move)
    if piece_type == 2:  # Laser
        return

    # For all other pieces, highlight adjacent empty cells
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for dr, dc in directions:
        new_r, new_c = row + dr, col + dc

        # Skip invalid positions or non-empty cells
        if not state.is_valid_position(new_r, new_c) or state.board[new_r, new_c, 0] != 0:
            continue

        # Draw highlight
        center_x = board_x + new_c * CELL_SIZE + CELL_SIZE // 2
        center_y = board_y + new_r * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, GREEN, (center_x, center_y), 10)


def draw_game_info(state, message=""):
    """Draw game information at the bottom of the screen"""
    info_rect = pygame.Rect(0, SCREEN_HEIGHT - INFO_HEIGHT, SCREEN_WIDTH, INFO_HEIGHT)
    pygame.draw.rect(screen, LIGHT_GRAY, info_rect)
    pygame.draw.line(screen, BLACK, (0, SCREEN_HEIGHT - INFO_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT - INFO_HEIGHT), 2)

    # Draw current player
    player_text = f"Player {state.current_player}'s Turn"
    player_color = PLAYER_COLORS[state.current_player]
    text_surf = font.render(player_text, True, player_color)
    screen.blit(text_surf, (20, SCREEN_HEIGHT - INFO_HEIGHT + 20))

    # Draw game status/message
    if message:
        msg_surf = font.render(message, True, BLACK)
        screen.blit(msg_surf, (20, SCREEN_HEIGHT - INFO_HEIGHT + 50))

    # Draw help text
    help_text = "Left-click: Select/Move | Right-click: Rotate CW | Middle-click: Rotate CCW | R: Reset | Q: Quit"
    help_surf = small_font.render(help_text, True, DARK_GRAY)
    screen.blit(help_surf, (20, SCREEN_HEIGHT - 25))


def draw_game_over(state):
    """Draw game over overlay"""
    # Semi-transparent overlay
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))

    # Game over text
    game_over_text = title_font.render("GAME OVER", True, WHITE)
    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50))

    # Winner text
    if state.winner == 0:
        winner_text = font.render("It's a draw!", True, WHITE)
    else:
        winner_color = PLAYER_COLORS[state.winner]
        winner_text = font.render(f"Player {state.winner} wins!", True, winner_color)

    screen.blit(winner_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2))

    # Play again text
    again_text = font.render("Press 'R' to play again or 'Q' to quit", True, WHITE)
    screen.blit(again_text, (SCREEN_WIDTH // 2 - 160, SCREEN_HEIGHT // 2 + 50))


def get_cell_from_pos(pos, board_x, board_y, state):
    """Convert mouse position to board cell coordinates"""
    x, y = pos

    # Check if click is within board bounds
    if (x < board_x or y < board_y or
            x >= board_x + state.cols * CELL_SIZE or
            y >= board_y + state.rows * CELL_SIZE):
        return None

    # Calculate row and column
    col = (x - board_x) // CELL_SIZE
    row = (y - board_y) // CELL_SIZE

    if row < 0 or row >= state.rows or col < 0 or col >= state.cols:
        return None

    return (row, col)


def check_rotate(state, pos, board_x, board_y, clockwise=True):
    """Check if we can rotate a piece and do so if valid"""
    cell = get_cell_from_pos(pos, board_x, board_y, state)
    if not cell:
        return state, False, "Invalid position"

    row, col = cell

    # Check if there's a piece belonging to current player
    if state.board[row, col, 1] != state.current_player:
        return state, False, "Not your piece"

    # Check if piece can be rotated (not a king or obelisk)
    piece_type = state.board[row, col, 0]
    if piece_type == 1 or piece_type == 5:  # King or Obelisk
        return state, False, f"{['King', '', '', '', '', 'Obelisk'][piece_type]} cannot be rotated"

    # Create the rotate action
    action_type = 1 if clockwise else 2  # 1=CW, 2=CCW
    action = Action(action_type, (row, col))

    # Apply the action
    new_state = state.apply_action(action)

    # Check if the action was valid
    if new_state.current_player == state.current_player and not new_state.game_over:
        return state, False, "Invalid action"

    return new_state, True, f"Rotated piece at ({row},{col})"


import pygame
import sys
from game_state import GameState, Action
from ai_player_learning import ai_player_learning  # <-- Import your AI function


def preload_ai_in_background():
    """Preload the AI in a background thread before the game starts"""
    global global_mcts
    print("Preloading AI model in background thread...")
    load_tree()
    if global_mcts is None:
        # Create a minimal state to initialize the tree
        state = GameState()
        adapter = LaserChessAdapter(state)
        global_mcts = PersistentMCTS(exploration_weight=1.4, time_limit=1.0)
        global_mcts.initialize_with_state(adapter)
    print("AI model preloaded successfully!")


def show_loading_screen():
    """Display a loading screen while AI is being initialized"""
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Chess - Loading...")

    font = pygame.font.SysFont("Arial", 24)
    title_font = pygame.font.SysFont("Arial", 36, bold=True)

    loading_text = title_font.render("Laser Chess", True, BLACK)
    subtitle_text = font.render("Loading AI model...", True, DARK_GRAY)

    clock = pygame.time.Clock()

    # Simple animation variables
    dots = 0
    frame = 0

    # Keep showing loading screen until AI is ready
    while global_mcts is None:
        screen.fill(WHITE)

        # Draw title
        screen.blit(loading_text, (SCREEN_WIDTH // 2 - loading_text.get_width() // 2,
                                   SCREEN_HEIGHT // 2 - 100))

        # Animated loading text
        if frame % 15 == 0:  # Update every 15 frames
            dots = (dots + 1) % 4

        loading_status = "Loading AI" + "." * dots
        status_text = font.render(loading_status, True, BLUE)
        screen.blit(status_text, (SCREEN_WIDTH // 2 - status_text.get_width() // 2,
                                  SCREEN_HEIGHT // 2))

        # Draw hint
        hint_text = small_font.render("AI is being loaded in the background. Please wait...", True, DARK_GRAY)
        screen.blit(hint_text, (SCREEN_WIDTH // 2 - hint_text.get_width() // 2,
                                SCREEN_HEIGHT // 2 + 100))

        pygame.display.flip()

        # Process events so the window doesn't freeze
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        frame += 1
        clock.tick(30)


def main():
    pygame.init()

    # Start preloading AI in a background thread
    loading_thread = threading.Thread(target=preload_ai_in_background)
    loading_thread.daemon = True
    loading_thread.start()

    # Show loading screen until AI is ready
    show_loading_screen()

    # Now continue with the regular game setup
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Chess")

    state = GameState()
    running = True
    selected_pos = None
    message = "Welcome to Laser Chess! Click a piece to begin."
    board_x, board_y = 0, 0


    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Chess")

    state = GameState()
    running = True
    selected_pos = None
    message = "Welcome to Laser Chess! Click a piece to begin."
    board_x, board_y = 0, 0

    clock = pygame.time.Clock()

    # For controlling when we’ve just finished a human move
    human_move_done = False

    while running:
        screen.fill(WHITE)

        # Draw the board
        board_x, board_y = draw_board(state)

        # If it's the human's piece that’s selected, draw valid moves
        if (selected_pos
                and state.board[selected_pos[0], selected_pos[1], 1] == 1
                and state.current_player == 1):
            draw_valid_moves(state, selected_pos, board_x, board_y)

        draw_laser_path(state, board_x, board_y)
        draw_pieces(state, board_x, board_y)

        if selected_pos:
            draw_selected_highlight(selected_pos[0], selected_pos[1], board_x, board_y)

        draw_game_info(state, message)

        # Inside main.py where it handles game over state:
        if state.game_over:
            draw_game_over(state)

            # Only save at the end of the game to avoid freezing during gameplay
            from ai_player_learning import save_tree, update_learning_stats

            # Update stats and save tree in a background thread to avoid freezing
            def save_game_data():
                update_learning_stats(state.winner)
                save_tree()

            # Run the save operation in a background thread
            save_thread = threading.Thread(target=save_game_data)
            save_thread.daemon = True
            save_thread.start()

        pygame.display.flip()

        # ------------------------------------------------------------------
        # If we just finished the human move, we do a small wait,
        # then let the AI move in the next iteration.
        # ------------------------------------------------------------------
        if human_move_done and not state.game_over and state.current_player == 2:
            pygame.time.wait(500)  # Pause so the user sees their move
            ai_action = ai_player_learning(state, time_limit=1.0)
            new_state = state.apply_action(ai_action)

            if new_state.current_player == state.current_player and not new_state.game_over:
                message = "⚠️ AI made an invalid move or got stuck!"
                print("[ERROR] AI move didn't change the game state.")
            else:
                state = new_state
                message = f"AI moved: {ai_action}"

            selected_pos = None
            human_move_done = False  # Reset the flag

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    state = GameState()
                    selected_pos = None
                    message = "Game reset. Player 1's turn."

            # ---------------------
            # Only handle clicks if:
            #   1) Game not over
            #   2) It's Player 1's turn
            # ---------------------
            elif (event.type == pygame.MOUSEBUTTONDOWN
                  and not state.game_over
                  and state.current_player == 1):
                pos = pygame.mouse.get_pos()

                if event.button == 1:  # Left click => select / move piece
                    cell = get_cell_from_pos(pos, board_x, board_y, state)
                    if cell:
                        row, col = cell
                        if selected_pos is None:
                            # Select a piece
                            if state.board[row, col, 1] == 1:  # Belongs to Player 1
                                selected_pos = (row, col)
                                message = f"Selected piece at ({row},{col})."
                            else:
                                message = "You can only select your own pieces."
                        else:
                            # Attempt a move
                            src_row, src_col = selected_pos
                            if (row, col) == selected_pos:
                                # Deselect if clicking the same piece
                                selected_pos = None
                                message = "Piece deselected."
                            elif state.board[row, col, 1] == 1:
                                # Switch to another piece
                                selected_pos = (row, col)
                                message = f"Selected piece at ({row},{col})."
                            else:
                                # Move attempt
                                piece_type = state.board[src_row, src_col, 0]
                                if piece_type == 2:
                                    message = "Lasers cannot move!"
                                else:
                                    action = Action(0, (src_row, src_col), (row, col))
                                    new_state = state.apply_action(action)
                                    if (new_state.current_player == state.current_player
                                            and not new_state.game_over):
                                        # Move invalid, same player still
                                        message = "Invalid move!"
                                    else:
                                        state = new_state
                                        message = f"Moved piece from ({src_row},{src_col}) to ({row},{col})."
                                        human_move_done = True
                                    selected_pos = None

                elif event.button == 3:  # Right click => rotate CW
                    new_state, success, msg = check_rotate(state, pos, board_x, board_y, clockwise=True)
                    if success:
                        state = new_state
                        selected_pos = None
                        human_move_done = True
                    message = msg

                elif event.button == 2:  # Middle click => rotate CCW
                    new_state, success, msg = check_rotate(state, pos, board_x, board_y, clockwise=False)
                    if success:
                        state = new_state
                        selected_pos = None
                        human_move_done = True
                    message = msg

        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()