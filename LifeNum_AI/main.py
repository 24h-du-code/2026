"""
main.py — launches all games in parallel, each on its own daemon thread.
Each game runs in an infinite loop (finish → new game → repeat).

Hackathon rule: only ONE active session per game type at a time.
→ Each thread handles one game ID sequentially.
"""
import threading
import stop_all_games
from modules.games import tic_tac_toe              # game 1
from modules.games import car_racing               # game 2
from modules.games import adaptive_traffic_racing  # game 5
from modules.games import partial_visibility_maze  # game 6
from modules.games import lava_maze                # game 7
from modules.games import key_door_maze            # game 8
from modules.games import lava_key_door_maze       # game 9
from modules.games import rush_hour                # game 4
from modules.games import snake                    # game 3
from modules.games import moon_lander              # game 10

GAME_RUNNERS = {
    1: tic_tac_toe.run,
    2: car_racing.run,
    3: snake.run,
    4: rush_hour.run,
    5: adaptive_traffic_racing.run,
    6: partial_visibility_maze.run,
    7: lava_maze.run,
    8: key_door_maze.run,
    9: lava_key_door_maze.run,
    10: moon_lander.run,
}


def launch_game_thread(game_id, run_fn):
    def target():
        print(f"[Thread game-{game_id}] Started.")
        try:
            run_fn()
        except Exception as e:
            print(f"[Thread game-{game_id}] Crashed: {e}")

    t = threading.Thread(target=target, name=f"game-{game_id}", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    stop_all_games.stopAllGames()

    threads = [launch_game_thread(gid, fn) for gid, fn in GAME_RUNNERS.items()]
    print(f"\n{len(threads)} game thread(s) running. Press Ctrl+C to stop.\n")

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nShutting down — stopping all active game sessions...")
        stop_all_games.stopAllGames()
        print("Done.")