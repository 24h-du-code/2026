"""
modules/games/snake.py — Snake agent (Game #3)

Strategy: Hamilton-cycle-inspired path that fills the board, falling back to
BFS-to-food when safe, with tail-chasing escape when trapped.

Manual override: if a file /tmp/plaiades_cmd_3.txt exists and contains a
valid action, that action is consumed once (the file is deleted) and used
instead of the AI decision.  The dashboard writes to this file via act.php.

Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.
"""
import json
import os
import time
import threading
from collections import deque
from ..endpoints import newGame, getState, act, stopGame

GAME_ID   = 3
BOARD     = 20        # 20×20 grid
WIN_LEN   = 40        # segments needed to win
CMD_FILE  = f"/tmp/plaiades_cmd_{GAME_ID}.txt"
VALID_ACTIONS = {"up", "down", "left", "right"}

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    name = threading.current_thread().name
    with _print_lock:
        print(f"[{name}]", *args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# API wrapper — sleep BEFORE every call to respect 1 call/sec rate limit
# ---------------------------------------------------------------------------

def is_api_error(resp):
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
# Manual override — read and consume one command from the command file
# ---------------------------------------------------------------------------

def pop_manual_command():
    """Return a manual action string if one was written, else None."""
    try:
        if not os.path.exists(CMD_FILE):
            return None
        with open(CMD_FILE, "r") as f:
            cmd = f.read().strip().lower()
        os.remove(CMD_FILE)
        if cmd in VALID_ACTIONS:
            return cmd
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Snake AI strategy
# ---------------------------------------------------------------------------

def state_to_sets(state):
    """Return (head, body_set, food) from API state."""
    snake = state.get("snake", [])           # list of [row, col]
    food  = tuple(state.get("food", [0, 0]))  # [row, col]
    if not snake:
        return None, set(), food
    head = tuple(snake[0])
    body = {tuple(s) for s in snake}
    return head, body, food


DIRS_MAP = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}
DIR_NAMES = list(DIRS_MAP.keys())


def neighbors(pos, body, exclude_tail=None):
    """Return valid (action, next_pos) pairs that don't collide."""
    r, c = pos
    result = []
    for action, (dr, dc) in DIRS_MAP.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < BOARD and 0 <= nc < BOARD:
            np = (nr, nc)
            if np not in body or np == exclude_tail:
                result.append((action, np))
    return result


def bfs_path(start, goal, body, max_steps=BOARD * BOARD):
    """BFS from start to goal avoiding body. Returns action list or None."""
    if start == goal:
        return []
    q = deque([(start, [])])
    visited = {start}
    steps = 0
    while q and steps < max_steps:
        pos, path = q.popleft()
        for action, nxt in neighbors(pos, body):
            if nxt == goal:
                return path + [action]
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [action]))
        steps += 1
    return None


def flood_fill_size(start, body):
    """Count reachable cells from start (body = blocked cells)."""
    visited = {start}
    q = deque([start])
    while q:
        pos = q.popleft()
        for _, nxt in neighbors(pos, body):
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return len(visited)


def is_safe_move(action, head, body, snake_list):
    """
    A move is 'safe' if the resulting position leaves enough open space
    (flood fill ≥ half the board) to keep playing.
    """
    dr, dc = DIRS_MAP[action]
    nxt = (head[0] + dr, head[1] + dc)
    if not (0 <= nxt[0] < BOARD and 0 <= nxt[1] < BOARD):
        return False
    if nxt in body and nxt != tuple(snake_list[-1]):
        return False
    new_body = body - {tuple(snake_list[-1])} | {nxt}
    return flood_fill_size(nxt, new_body) >= (BOARD * BOARD) // 4


def choose_action(state):
    """
    Decision priority:
    1. If path to food is safe AND eating doesn't trap us → go to food.
    2. Chase tail (longest safe path) to buy time.
    3. Any safe neighbor by flood-fill.
    4. Least-bad neighbor (furthest from body).
    """
    snake_list = state.get("snake", [])
    food_raw   = state.get("food", [0, 0])
    if not snake_list:
        return "right"

    head  = tuple(snake_list[0])
    tail  = tuple(snake_list[-1])
    body  = {tuple(s) for s in snake_list}
    food  = tuple(food_raw)

    # 1. Try BFS to food
    path = bfs_path(head, food, body)
    if path:
        action = path[0]
        # Simulate eating and check if tail is still reachable (safe)
        dr, dc = DIRS_MAP[action]
        new_head = (head[0] + dr, head[1] + dc)
        new_body = body | {new_head}           # food eaten → body grows
        tail_reachable = bfs_path(new_head, tail, new_body) is not None
        space_ok = flood_fill_size(new_head, new_body) >= len(snake_list) + 2
        if tail_reachable and space_ok:
            return action

    # 2. Chase tail (keeps snake compact, avoids trapping itself)
    tail_path = bfs_path(head, tail, body)
    if tail_path:
        return tail_path[0]

    # 3. Any safe move by flood fill
    safe = [(a, n) for a, n in neighbors(head, body)
            if is_safe_move(a, head, body, snake_list)]
    if safe:
        # Prefer the direction with most open space
        return max(safe, key=lambda x: flood_fill_size(x[1], body - {tail} | {x[1]}))[0]

    # 4. Fallback: any valid move
    valid = neighbors(head, body, exclude_tail=tail)
    if valid:
        return valid[0][0]

    return "right"  # absolute last resort


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_game():
    tprint("=== New game ===")
    new_resp = safe_call(newGame, GAME_ID)
    session_id = new_resp["gamesessionid"]
    tprint(f"Session {session_id}")

    status = "continue"
    step   = 0

    while status == "continue":
        state_resp = safe_call(getState, session_id)
        state = state_resp.get("state", state_resp)

        if state.get("done"):
            break

        # Emit full state for canvas renderer
        snake = state.get("snake", [])
        food  = state.get("food", [0, 0])
        score = state.get("score", len(snake))
        tprint(f"STATE_JSON:{json.dumps({'snake': snake, 'food': food, 'score': score})}")

        # Manual override takes priority over AI
        manual_cmd = pop_manual_command()
        if manual_cmd:
            action = manual_cmd
            tprint(f"[MANUAL] step={step} score={score} → {action}")
        else:
            action = choose_action(state)
            tprint(f"step={step} score={score} → {action}")

        act_resp = safe_call(act, session_id, action)
        status = act_resp.get("status", "continue")

        # Update state from act response for next iteration
        if "state" in act_resp:
            updated = act_resp["state"]
            snake = updated.get("snake", snake)
            food  = updated.get("food", food)
            score = updated.get("score", score)
            tprint(f"STATE_JSON:{json.dumps({'snake': snake, 'food': food, 'score': score})}")

        step += 1

    tprint(f"Result: {status.upper()}")
    return status


# ---------------------------------------------------------------------------
# Infinite runner (called by main.py thread)
# ---------------------------------------------------------------------------

def run():
    results = {"win": 0, "lose": 0, "other": 0}
    game_num = 0

    # Clean up any stale command file from a previous run
    try:
        os.remove(CMD_FILE)
    except OSError:
        pass

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
        net = results["win"] * 3 - results["lose"]
        tprint(f"Totals — W:{results['win']} L:{results['lose']} | Score: {net:+d}")

        time.sleep(1)


if __name__ == "__main__":
    run()