"""
modules/games/car_racing.py — Car Racing agent.

Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.

Strategy: BFS lookahead over the next LOOKAHEAD steps to pick the lane
with the most reachable futures, avoiding obstacles.
"""
import json
import time
import threading
from ..endpoints import newGame, getState, act, stopGame

GAME_ID   = 2
LOOKAHEAD = 8

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
# Strategy
# ---------------------------------------------------------------------------

def obstacle_set(obs):
    return {(o["step"], o["lane"]) for o in obs}


def reachable_futures(start_lane, start_step, depth, obs_set):
    """Count total reachable lane-slots over the next `depth` steps."""
    current = {start_lane}
    total = 0
    for step in range(start_step, start_step + depth):
        nxt = set()
        for l in current:
            for nl in (l - 1, l, l + 1):
                if 0 <= nl <= 2 and (step + 1, nl) not in obs_set:
                    nxt.add(nl)
        total += len(nxt)
        current = nxt
        if not current:
            break
    return total


def choose_action(state):
    pos      = state["position"]
    lane     = state["lane"]
    obs_list = state.get("upcoming_obstacles", state.get("obstacles", []))
    obs_set  = obstacle_set(obs_list)
    next_pos = pos + 1

    moves = {"stay": lane, "move_left": lane - 1, "move_right": lane + 1}
    moves = {a: l for a, l in moves.items() if 0 <= l <= 2}

    scores = {}
    for action, target_lane in moves.items():
        if (next_pos, target_lane) in obs_set:
            scores[action] = -1  # immediate crash — discard
        else:
            future = reachable_futures(target_lane, next_pos, LOOKAHEAD, obs_set)
            # slight penalty for moving to avoid unnecessary lane changes
            scores[action] = (future, 0 if action == "stay" else -1)

    viable = {a: s for a, s in scores.items() if s != -1}
    if viable:
        return max(viable, key=lambda a: viable[a])

    # All moves crash — pick lane with the furthest obstacle
    def closest_obstacle(l):
        dists = [o["step"] - next_pos for o in obs_list
                 if o["lane"] == l and o["step"] >= next_pos]
        return min(dists) if dists else 999

    return max(moves, key=lambda a: (closest_obstacle(moves[a]),
                                     1 if a == "stay" else 0))


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

        action = choose_action(state)
        tprint(f"pos={state.get('position')} lane={state.get('lane')} → {action}")
        tprint(f"STATE_JSON:{json.dumps({'position': state.get('position', 0), 'lane': state.get('lane', 1), 'upcoming_obstacles': state.get('upcoming_obstacles', state.get('obstacles', []))})}")

        act_resp = safe_call(act, session_id, action)
        status = act_resp.get("status", "continue")

    tprint(f"Result: {status.upper()}")
    return status


# ---------------------------------------------------------------------------
# Infinite runner (called by main.py thread)
# ---------------------------------------------------------------------------

def run():
    results = {"win": 0, "lose": 0, "other": 0}
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
        net = results["win"] * 2 - results["lose"]
        tprint(f"Totals — W:{results['win']} L:{results['lose']} | Score: {net:+d}")

        time.sleep(1)


if __name__ == "__main__":
    run()