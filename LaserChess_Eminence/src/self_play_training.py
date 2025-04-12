import os
import time
import random
import json
import sys
import multiprocessing
from datetime import datetime
from game_state import GameState, Action
from ai_player_learning import ai_player_learning, save_tree, reset_ai, load_tree, get_learning_stats

# Directory for training logs
TRAINING_LOGS_DIR = "training_logs"


def ensure_logs_dir():
    """Make sure the training logs directory exists"""
    if not os.path.exists(TRAINING_LOGS_DIR):
        os.makedirs(TRAINING_LOGS_DIR)


def log_training_results(results):
    """Log training results to a JSON file"""
    ensure_logs_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(TRAINING_LOGS_DIR, f"training_log_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📊 Training results saved to {filename}")


def run_self_play_games(num_games=50, time_limit=1.0, save_interval=5, curriculum=True, max_moves=150):
    """
    Run multiple self-play games to train the AI with enhanced learning.

    Parameters:
    - num_games: Total games to run
    - time_limit: Base time allocated for each MCTS decision
    - save_interval: Save the tree every N games
    - curriculum: Whether to use curriculum learning (gradually increase difficulty)
    - max_moves: Maximum number of moves before ending a game (to prevent infinite games)
    """
    print(f"🔁 Starting self-play training: {num_games} games")

    training_results = {
        "start_time": datetime.now().isoformat(),
        "num_games": num_games,
        "base_time_limit": time_limit,
        "games": []
    }

    # Try to load existing tree first
    load_tree()

    # Get starting AI stats
    starting_stats = get_learning_stats()
    training_results["starting_stats"] = starting_stats

    # Track win rates over time to adjust difficulty
    recent_wins = []

    for game_num in range(1, num_games + 1):
        print(f"\n🎮 Game {game_num}/{num_games}")
        game_start_time = time.time()

        # Create fresh game state
        state = GameState()

        # For curriculum learning, adjust parameters based on performance
        current_time_limit = time_limit
        exploration_bonus = 0.0

        if curriculum and recent_wins:
            # Calculate win rate over last 10 games (or fewer if we don't have 10 yet)
            win_rate = sum(recent_wins[-10:]) / len(recent_wins[-10:]) if recent_wins else 0.5

            # Adjust parameters based on win rate
            if win_rate > 0.7:  # AI is doing too well, make it harder
                current_time_limit *= 0.8  # Reduce thinking time
                exploration_bonus = -0.2  # Less exploration
                print(f"🔼 Increasing difficulty: time={current_time_limit:.2f}s, exploration={exploration_bonus}")
            elif win_rate < 0.3:  # AI is struggling, make it easier
                current_time_limit *= 1.2  # More thinking time
                exploration_bonus = 0.2  # More exploration
                print(f"🔽 Decreasing difficulty: time={current_time_limit:.2f}s, exploration={exploration_bonus}")

        # Track game events
        game_events = []
        move_count = 0

        try:
            # Play the game with a move limit to prevent infinite games
            while not state.game_over and move_count < max_moves:
                move_start = time.time()

                # Get action from the AI
                action = ai_player_learning(
                    state,
                    time_limit=current_time_limit,
                    max_simulations=1000
                )

                if action is None:
                    print("[ERROR] AI couldn't find any valid moves. Game is stuck. Ending game.")
                    state.game_over = True
                    state.winner = 0  # Force a draw
                    break

                move_duration = time.time() - move_start

                # Apply the action
                new_state = state.apply_action(action)

                # Log the move
                game_events.append({
                    "move_number": move_count + 1,
                    "player": state.current_player,
                    "action": str(action),
                    "thinking_time": move_duration
                })

                # Check if the move was valid
                if new_state.current_player == state.current_player and not new_state.game_over:
                    print(f"[⚠️] AI made an invalid move. Game stuck. Skipping.")
                    state.game_over = True
                    state.winner = 3 - state.current_player  # The other player wins
                    break

                # Update the state and move counter
                state = new_state
                move_count += 1

                # Display occasional progress
                if move_count % 5 == 0:
                    print(f"  Move {move_count}, current player: {state.current_player}")

                # Check if we reached the move limit
                if move_count >= max_moves:
                    print(f"[⚠️] Game reached maximum move limit ({max_moves}). Ending game.")
                    # Force a draw
                    state.game_over = True
                    state.winner = 0

            # Game complete
            game_duration = time.time() - game_start_time
            print(f"🏁 Game {game_num} ended after {move_count} moves ({game_duration:.1f}s)")
            print(f"  Winner: Player {state.winner}")

            # Update recent wins tracking (for curriculum)
            if state.winner == 2:  # AI won as player 2
                recent_wins.append(1)
            else:
                recent_wins.append(0)

            # Keep recent wins list manageable
            if len(recent_wins) > 20:
                recent_wins = recent_wins[-20:]

            # Save the game result
            game_result = {
                "game_number": game_num,
                "moves": move_count,
                "winner": state.winner,
                "duration_seconds": game_duration,
                "time_limit_used": current_time_limit,
                "events": game_events
            }
            training_results["games"].append(game_result)

        except Exception as e:
            print(f"[❌] Error in game {game_num}: {e}")
            # Don't let one failed game ruin the whole training session
            game_result = {
                "game_number": game_num,
                "error": str(e),
                "moves": move_count
            }
            training_results["games"].append(game_result)

        # Save tree periodically
        if game_num % save_interval == 0:
            save_tree()
            print(f"💾 Saved AI tree at game {game_num}")

    # Final save
    save_tree()

    # Complete training data
    training_results["end_time"] = datetime.now().isoformat()
    training_results["final_stats"] = get_learning_stats()

    # Calculate summary statistics
    total_moves = sum(game["moves"] for game in training_results["games"] if "moves" in game)
    ai_wins = sum(1 for game in training_results["games"] if game.get("winner") == 2)

    training_results["summary"] = {
        "total_games_completed": len(training_results["games"]),
        "total_moves": total_moves,
        "ai_win_percentage": ai_wins / len(training_results["games"]) * 100 if training_results["games"] else 0
    }

    # Log the results
    log_training_results(training_results)

    print("✅ Self-play training complete. Final tree saved.")
    print(
        f"📈 AI win rate: {ai_wins}/{len(training_results['games'])} ({ai_wins / len(training_results['games']) * 100:.1f}%)")


def run_parallel_self_play(num_games=10, time_limit=2.0, save_interval=5, max_moves=150):
    """Run multiple self-play games in parallel to speed up training"""
    global learning_stats  # Add this line

    # Determine number of processes based on CPU cores
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores - 1, 4)  # Use n-1 cores, max 4 to avoid overloading

    print(f"🚀 Starting parallel self-play with {num_processes} processes")

    # Create a shared queue for results
    manager = multiprocessing.Manager()
    results_queue = manager.Queue()

    # Divide games among processes
    games_per_process = num_games // num_processes
    remaining_games = num_games % num_processes

    processes = []

    for i in range(num_processes):
        # Distribute remaining games
        process_games = games_per_process + (1 if i < remaining_games else 0)

        # Create and start process
        process = multiprocessing.Process(
            target=_run_process_games,
            args=(i, process_games, time_limit, save_interval, max_moves, results_queue)
        )
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Merge and save final results
    print("⚙️ Merging results from all processes")
    merged_results = {
        "start_time": datetime.now().isoformat(),
        "num_games": num_games,
        "base_time_limit": time_limit,
        "parallel_processes": num_processes,
        "games": []
    }

    # Merge the process results and update global learning stats
    total_games_played = 0
    total_wins = 0

    for result in results:
        merged_results["games"].extend(result["games"])

        # Update counts from each process
        if "learning_stats" in result:
            proc_stats = result["learning_stats"]
            total_games_played += proc_stats.get("games_played", 0)
            total_wins += proc_stats.get("wins", 0)

    # Update global learning stats
    learning_stats["games_played"] += total_games_played
    learning_stats["wins"] += total_wins
    learning_stats["moves_made"] += sum(len(r["games"]) for r in results)

    # Calculate summary statistics
    total_moves = sum(game["moves"] for game in merged_results["games"] if "moves" in game)
    ai_wins = sum(1 for game in merged_results["games"] if game.get("winner") == 2)

    merged_results["summary"] = {
        "total_games_completed": len(merged_results["games"]),
        "total_moves": total_moves,
        "ai_win_percentage": ai_wins / len(merged_results["games"]) * 100 if merged_results["games"] else 0
    }

    # Save the main tree with updated stats
    save_tree()

    # Log the final results
    log_training_results(merged_results)

    print("✅ Parallel self-play training complete.")
    print(
        f"📈 Overall AI win rate: {ai_wins}/{len(merged_results['games'])} ({ai_wins / len(merged_results['games']) * 100:.1f}%)")
    print(f"📊 AI has now played {learning_stats['games_played']} total games with {learning_stats['wins']} wins")


def _run_process_games(process_id, num_games, time_limit, save_interval, max_moves, results_queue):
    """Run a set of games in a single process"""
    print(f"Process {process_id}: Starting {num_games} games")

    # Each process gets its own tree file
    tree_file = os.path.join("ai_knowledge", f"mcts_tree_process_{process_id}.pkl")

    # Set up process-specific results
    process_results = {
        "process_id": process_id,
        "num_games": num_games,
        "games": []
    }

    # Track games won by this process
    wins = 0
    games_completed = 0

    # Run games
    for game_num in range(1, num_games + 1):
        try:
            # Create fresh game state
            state = GameState()
            game_start_time = time.time()

            # Track game events
            game_events = []
            move_count = 0

            # Play the game
            while not state.game_over and move_count < max_moves:
                # Get action from the AI
                action = ai_player_learning(state, time_limit=time_limit)

                if action is None:
                    break

                # Apply the action
                new_state = state.apply_action(action)

                # Check if the move was valid
                if new_state.current_player == state.current_player and not new_state.game_over:
                    break  # Game stuck

                # Update state and counter
                state = new_state
                move_count += 1

            # Game complete
            games_completed += 1
            game_duration = time.time() - game_start_time

            # Track wins
            if state.winner == 2:  # AI wins as player 2
                wins += 1

            # Save the game result
            game_result = {
                "process_id": process_id,
                "game_number": game_num,
                "moves": move_count,
                "winner": state.winner,
                "duration_seconds": game_duration
            }
            process_results["games"].append(game_result)

            # Periodically save tree
            if game_num % save_interval == 0:
                save_tree(tree_file)

        except Exception as e:
            # Log error but continue
            print(f"Process {process_id}, Game {game_num}: Error - {e}")
            process_results["games"].append({
                "process_id": process_id,
                "game_number": game_num,
                "error": str(e)
            })

    # Add learning stats to results
    process_results["learning_stats"] = {
        "games_played": games_completed,
        "wins": wins
    }

    # Put results in the shared queue
    results_queue.put(process_results)
    print(f"Process {process_id}: Completed {num_games} games")


def run_curriculum_training():
    """
    Run a structured curriculum of training stages.
    Each stage increases in difficulty and builds on the previous knowledge.
    """
    print("🎓 Starting curriculum training sequence")

    # Increase recursion limit
    sys.setrecursionlimit(10000)

    # Stage 1: Initial training with high exploration
    print("\n--- Stage 1: Initial Exploration ---")
    run_self_play_games(
        num_games=10,
        time_limit=2.0,  # Moderate time limit for basic learning
        save_interval=5,
        curriculum=False,  # Fixed parameters for this stage
        max_moves=150  # Limit games to 150 moves
    )

    # Stage 2: Intermediate training with balanced parameters
    print("\n--- Stage 2: Building Knowledge ---")
    run_self_play_games(
        num_games=15,
        time_limit=3.0,  # Longer time for better decisions
        save_interval=5,
        curriculum=True,  # Start adapting difficulty
        max_moves=150  # Limit games to 150 moves
    )

    # Stage 3: Advanced training with longer thinking time
    print("\n--- Stage 3: Advanced Strategies ---")
    run_self_play_games(
        num_games=15,
        time_limit=5.0,  # Significantly longer time for deep search
        save_interval=5,
        curriculum=True,  # Continue adapting difficulty
        max_moves=150  # Limit games to 150 moves
    )

    print("\n🎯 Curriculum training complete!")


def run_accelerated_training():
    """
    Run an accelerated training regimen using parallel processing
    for faster completion while maintaining quality.
    """
    print("🚀 Starting accelerated training sequence")

    # Stage 1: Quick parallel training with moderate time
    print("\n--- Stage 1: Parallel Exploration ---")
    run_parallel_self_play(
        num_games=20,
        time_limit=1.5,
        save_interval=5,
        max_moves=100
    )

    # Stage 2: Deeper single-thread training for quality
    print("\n--- Stage 2: Deep Quality Learning ---")
    run_self_play_games(
        num_games=10,
        time_limit=4.0,  # Longer time for higher quality decisions
        save_interval=2,
        curriculum=True,
        max_moves=150
    )

    print("\n🏁 Accelerated training complete!")


if __name__ == "__main__":
    # Uncomment one of these options:

    # Option 1: Reset AI and start fresh (uncomment to use)
    #reset_ai(clear_files=True)

    # Option 2: Standard curriculum training (recommended for first-time training)
    run_curriculum_training()

    # Option 3: Faster training using parallelization
    #run_accelerated_training()

    # Option 4: Single stage training (for quick tests or specific improvements)
    #run_self_play_games(num_games=5, time_limit=3.0, curriculum=True)


