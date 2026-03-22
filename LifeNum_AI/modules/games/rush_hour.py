"""
modules/games/rush_hour.py — Rush Hour agent (Game #4)

BFS solver: finds the sequence of moves to free the red car (X) to the exit.
Stops immediately if the puzzle is unsolvable (avoids losing points).
Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.
"""
import json
import time
import threading
from collections import deque
from ..endpoints import newGame, getState, act, stopGame

GAME_ID        = 4
MAX_BFS_STATES = 500_000

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    name = threading.current_thread().name
    with _print_lock:
        print(f"[{name}]", *args, **kwargs)


# ---------------------------------------------------------------------------
# API wrapper — sleep BEFORE every call to respect 1 call/sec rate limit
# Handles: 429 rate limit, 503 service unavailable, missing keys
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
# Rush Hour BFS solver
# ---------------------------------------------------------------------------

def build_grid(vd, gs=6):
    g = [[None] * gs for _ in range(gs)]
    for vid, v in vd.items():
        for i in range(v['length']):
            r = v['row'] + (i if v['orientation'] == 'v' else 0)
            c = v['col'] + (i if v['orientation'] == 'h' else 0)
            if 0 <= r < gs and 0 <= c < gs:
                g[r][c] = vid
    return g


def get_moves(vd, gs=6, red_id='X', exit_row=3):
    g = build_grid(vd, gs)
    moves = []
    for vid, v in vd.items():
        row, col, o, l = v['row'], v['col'], v['orientation'], v['length']
        if o == 'h':
            if col > 0 and g[row][col - 1] is None:
                nv = dict(vd); nv[vid] = dict(v, col=col - 1)
                moves.append((f"move_{vid}_left", nv))
            is_exit = (vid == red_id and row == exit_row and col + l == gs)
            normal  = (col + l < gs and g[row][col + l] is None)
            if normal or is_exit:
                nv = dict(vd); nv[vid] = dict(v, col=col + 1)
                moves.append((f"move_{vid}_right", nv))
        else:
            if row > 0 and g[row - 1][col] is None:
                nv = dict(vd); nv[vid] = dict(v, row=row - 1)
                moves.append((f"move_{vid}_up", nv))
            if row + l < gs and g[row + l][col] is None:
                nv = dict(vd); nv[vid] = dict(v, row=row + 1)
                moves.append((f"move_{vid}_down", nv))
    return moves


def sk(vd):
    return frozenset((vid, v['row'], v['col']) for vid, v in vd.items())


def is_solved(vd, rid, er, gs=6):
    v = vd[rid]
    return v['row'] == er and v['col'] + v['length'] > gs


def solve(vehicles, red='X', exit_row=3, gs=6):
    vd0 = {v['id']: {'id': v['id'], 'row': v['pos'][0], 'col': v['pos'][1],
                      'orientation': v['orientation'], 'length': v['length']}
           for v in vehicles}
    if is_solved(vd0, red, exit_row, gs):
        return []
    q = deque([(vd0, [])])
    vis = {sk(vd0)}
    while q:
        if len(vis) > MAX_BFS_STATES:
            return None  # too complex — unsolvable within limit
        vd, acts = q.popleft()
        for action, nv in get_moves(vd, gs, red, exit_row):
            nk = sk(nv)
            if nk in vis:
                continue
            vis.add(nk)
            na = acts + [action]
            if is_solved(nv, red, exit_row, gs):
                return na
            q.append((nv, na))
    return None  # no solution exists


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_game():
    tprint("=== New game ===")
    new_resp = safe_call(newGame, GAME_ID)
    session_id = new_resp["gamesessionid"]
    tprint(f"Session {session_id}")

    state_resp = safe_call(getState, session_id)
    state = state_resp.get("state", state_resp)

    if state.get("done"):
        return "already_done"

    vehicles = state["vehicles"]
    exit_row = state["exit_pos"][0]
    red_id   = state.get("red_car_id", "X")
    gs       = state.get("grid_size", 6)

    # Emit initial board state for dashboard rendering
    tprint(f"STATE_JSON:{json.dumps({'vehicles': vehicles, 'grid_size': gs, 'exit_pos': state['exit_pos'], 'steps': 0})}")

    tprint(f"Solving... ({len(vehicles)} vehicles)")
    solution = solve(vehicles, red_id, exit_row, gs)

    if solution is None or (len(solution) == 0 and not state.get("done")):
        tprint("No solution found — stopping game.")
        safe_call(stopGame, session_id)
        return "stopped"

    tprint(f"Solution found: {len(solution)} moves")
    status = "continue"

    # Track vehicle positions through the solution for live rendering
    vd_current = {v['id']: {'id': v['id'], 'row': v['pos'][0], 'col': v['pos'][1],
                             'orientation': v['orientation'], 'length': v['length']}
                  for v in vehicles}

    for i, action in enumerate(solution):
        tprint(f"[{i+1}/{len(solution)}] {action}")
        # Apply the move to vd_current for dashboard rendering
        parts = action.split('_')  # move_X_right → ['move', 'X', 'right']
        if len(parts) == 3:
            vid, direction = parts[1], parts[2]
            if vid in vd_current:
                v = dict(vd_current[vid])
                if direction == 'left':  v['col'] -= 1
                elif direction == 'right': v['col'] += 1
                elif direction == 'up':  v['row'] -= 1
                elif direction == 'down': v['row'] += 1
                vd_current[vid] = v
        # Emit updated state: convert back to vehicles list format
        veh_list = [{'id': v['id'], 'pos': [v['row'], v['col']],
                     'orientation': v['orientation'], 'length': v['length']}
                    for v in vd_current.values()]
        tprint(f"STATE_JSON:{json.dumps({'vehicles': veh_list, 'grid_size': gs, 'exit_pos': state['exit_pos'], 'steps': i+1})}")
        act_resp = safe_call(act, session_id, action)
        status = act_resp.get("status", "continue")
        if status != "continue":
            break

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
        net = results["win"] * 4 - results["lose"]
        tprint(f"Totals — W:{results['win']} L:{results['lose']} S:{results['stopped']} | Score: {net:+d}")

        time.sleep(1)


if __name__ == "__main__":
    run()