"""
modules/games/partial_visibility_maze.py — Partial Visibility Maze agent (Game #6)

Frontier exploration + BFS to exit once revealed.
Vision radius: 2 cells around player (5x5 window).
Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.
"""
import json
import time
import threading
from collections import deque
from ..endpoints import newGame, getState, act, stopGame

GAME_ID    = 6
ROWS, COLS = 10, 10
DIRS       = [(0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right")]

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
# Map helpers
# ---------------------------------------------------------------------------

def empty_map():
    return [['?' for _ in range(COLS)] for _ in range(ROWS)]


def merge(gm, sg, pc, pr):
    """Merge the partial visibility grid into our global map."""
    for r in range(max(0, pr - 2), min(ROWS, pr + 3)):
        for c in range(max(0, pc - 2), min(COLS, pc + 3)):
            if sg[r][c] != '?':
                gm[r][c] = sg[r][c]


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

def _bfs(gm, sc, sr, gc, gr, allow_unknown):
    if (sc, sr) == (gc, gr):
        return []
    q = deque([((sc, sr), [])])
    vis = {(sc, sr)}
    while q:
        (col, row), path = q.popleft()
        for dc, dr, action in DIRS:
            nc, nr = col + dc, row + dr
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if (nc, nr) in vis:
                continue
            cell = gm[nr][nc]
            if cell in ('#', 'L'):
                continue
            if cell == '?' and not allow_unknown:
                continue
            np = path + [action]
            if (nc, nr) == (gc, gr):
                return np
            vis.add((nc, nr))
            q.append(((nc, nr), np))
    return None


def bfs_to_exit(gm, pc, pr, ec, er):
    """Try known path first, fall back to path through unknowns."""
    return _bfs(gm, pc, pr, ec, er, False) or _bfs(gm, pc, pr, ec, er, True)


def bfs_to_frontier(gm, pc, pr):
    """Navigate to the nearest unexplored ('?') cell."""
    q = deque([((pc, pr), [])])
    vis = {(pc, pr)}
    while q:
        (col, row), path = q.popleft()
        for dc, dr, action in DIRS:
            nc, nr = col + dc, row + dr
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if (nc, nr) in vis:
                continue
            cell = gm[nr][nc]
            vis.add((nc, nr))
            if cell == '?':
                return path + [action]
            if cell not in ('#', 'L'):
                q.append(((nc, nr), path + [action]))
    return None


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_game():
    tprint("=== New game ===")
    new_resp = safe_call(newGame, GAME_ID)
    session_id = new_resp["gamesessionid"]
    tprint(f"Session {session_id}")

    gm       = empty_map()
    exit_pos = None
    status   = "continue"

    while status == "continue":
        state_resp = safe_call(getState, session_id)
        state = state_resp.get("state", state_resp)

        if state.get("done"):
            break

        pc, pr = state["player_pos"]
        merge(gm, state["grid"], pc, pr)

        if state.get("exit_pos"):
            exit_pos = state["exit_pos"]

        # Emit accumulated map state for dashboard (player_pos uses [col, row] = [pc, pr])
        tprint(f"STATE_JSON:{json.dumps({'grid': gm, 'player_pos': [pc, pr], 'exit_pos': exit_pos, 'has_key': False, 'steps': state.get('steps', 0)})}")

        if exit_pos:
            ec, er = exit_pos
            path = bfs_to_exit(gm, pc, pr, ec, er)
            mode = "→exit"
        else:
            path = bfs_to_frontier(gm, pc, pr)
            mode = "→frontier"

        if path is None:
            tprint("No path found — stopping game.")
            safe_call(stopGame, session_id)
            return "stopped"

        if len(path) == 0:
            break

        action = path[0]
        tprint(f"pos=({pc},{pr}) {mode} → {action} (path len={len(path)})")

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
        net = results["win"] * 7 - results["lose"]
        tprint(f"Totals — W:{results['win']} L:{results['lose']} S:{results['stopped']} | Score: {net:+d}")


if __name__ == "__main__":
    run()
