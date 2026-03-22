"""
modules/games/key_door_maze.py — Key & Door Maze agent (Game #8)

Two-phase BFS: player → key → exit.
Stops immediately if no path exists (avoids losing points).
Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.
"""
import json
import time
import threading
from collections import deque
from ..endpoints import newGame, getState, act, stopGame

GAME_ID = 8
DIRS = [(0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right")]

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    name = threading.current_thread().name
    with _print_lock:
        print(f"[{name}]", *args, **kwargs)


# ---------------------------------------------------------------------------
# API wrapper — sleep BEFORE every call to respect 1 call/sec rate limit
# ---------------------------------------------------------------------------

def is_api_error(resp):
    """Return (is_error, reason) for any known transient API error."""
    if not isinstance(resp, dict):
        return True, "non-dict response"
    if "error" in resp:
        return True, resp["error"]
    if "message" in resp and "unavailable" in str(resp.get("message", "")).lower():
        return True, resp["message"]
    return False, None


def safe_call(fn, *args, retries=10, base_delay=1.0):
    for attempt in range(retries):
        time.sleep(base_delay)
        resp = fn(*args)
        err, reason = is_api_error(resp)
        if err:
            wait = base_delay * (attempt + 1)
            tprint(f"[API error] {reason} — waiting {wait:.1f}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        else:
            return resp
    raise RuntimeError(f"API unavailable after {retries} retries.")


# ---------------------------------------------------------------------------
# BFS — avoids walls (#) and lava (L); respects locked door (D)
# ---------------------------------------------------------------------------

def find_symbol(grid, sym):
    for ri, row in enumerate(grid):
        for ci, cell in enumerate(row):
            if cell == sym:
                return ci, ri
    return None


def bfs(grid, sc, sr, gc, gr, has_key):
    rows, cols = len(grid), len(grid[0])
    if (sc, sr) == (gc, gr):
        return []
    q = deque([((sc, sr), [])])
    vis = {(sc, sr)}
    while q:
        (col, row), path = q.popleft()
        for dc, dr, action in DIRS:
            nc, nr = col + dc, row + dr
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nc, nr) in vis:
                continue
            cell = grid[nr][nc]
            if cell == '#' or cell == 'L':
                continue
            if cell == 'D' and not has_key:
                continue
            np = path + [action]
            if (nc, nr) == (gc, gr):
                return np
            vis.add((nc, nr))
            q.append(((nc, nr), np))
    return None  # no path exists


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_game():
    tprint("=== New game ===")
    new_resp = safe_call(newGame, GAME_ID)
    session_id = new_resp["gamesessionid"]
    tprint(f"Session {session_id}")

    status = "continue"

    while status == "continue":
        state_resp = safe_call(getState, session_id)
        state = state_resp.get("state", state_resp)

        if state.get("done"):
            break

        grid         = state["grid"]
        pcol, prow   = state["player_pos"]
        ecol, erow   = state["exit_pos"]
        has_key      = state["has_key"]

        tprint(f"STATE_JSON:{json.dumps({'grid': grid, 'player_pos': [pcol, prow], 'exit_pos': [ecol, erow], 'has_key': has_key, 'steps': state.get('steps', 0)})}")

        # Phase 1: go to key if we don't have it yet
        # Phase 2: go to exit (door is now passable)
        if not has_key:
            kpos = find_symbol(grid, "K")
            if kpos:
                path = bfs(grid, pcol, prow, kpos[0], kpos[1], False)
            else:
                # Key already picked up but state not updated yet — head to exit
                path = bfs(grid, pcol, prow, ecol, erow, True)
        else:
            path = bfs(grid, pcol, prow, ecol, erow, True)

        if path is None:
            tprint("No safe path — stopping game.")
            safe_call(stopGame, session_id)
            return "stopped"

        if len(path) == 0:
            break

        action = path[0]
        phase = "→key" if not has_key else "→exit"
        tprint(f"pos=({pcol},{prow}) {phase} → {action} (path len={len(path)})")

        act_resp = safe_call(act, session_id, action)
        status = act_resp.get("status", "continue")

    tprint(f"Result: {status.upper()}")
    return status


# ---------------------------------------------------------------------------
# Infinite runner (called by main.py thread)
# ---------------------------------------------------------------------------

def run():
    results = {"win": 0, "lose": 0, "stopped": 0, "other": 0}
    game_num = 0

    while True:
        game_num += 1
        tprint(f"--- Game #{game_num} ---")
        try:
            outcome = play_game()
        except Exception as e:
            tprint(f"[ERROR] {e}")
            outcome = "other"
            time.sleep(2)

        results[outcome if outcome in results else "other"] += 1
        net = results["win"] * 9 - results["lose"]
        tprint(f"Totals — W:{results['win']} L:{results['lose']} S:{results['stopped']} | Score: {net:+d}")


if __name__ == "__main__":
    run()
