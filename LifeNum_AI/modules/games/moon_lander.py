"""
modules/games/moon_lander.py — Noisy Moon Lander Lite agent (Game #10)

Strategy: PD controller on position, velocity and tilt.
  - Horizontal: correct dx_pad with left/right rotation
  - Vertical:   fire main thrust to counteract gravity when above pad
  - Tilt:       use stabilize to zero angular velocity
  - Landing:    throttle down gently as altitude approaches zero

Manual override: if /tmp/plaiades_cmd_10.txt exists and contains a valid
action, that action is consumed once and used instead of the controller.
The dashboard writes to this file via act.php.

Actions: idle, main, left, right, main_left, main_right, stabilize
Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.
"""
import json
import math
import os
import time
import threading
from ..endpoints import newGame, getState, act, stopGame

GAME_ID       = 10
CMD_FILE      = f"/tmp/plaiades_cmd_{GAME_ID}.txt"
VALID_ACTIONS = {"idle", "main", "left", "right", "main_left", "main_right", "stabilize"}

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
# PD controller — reads observation_schema if available, falls back to
# direct state field names from the API doc
# ---------------------------------------------------------------------------

def extract_obs(state):
    """
    Extract named features from state.
    Uses observation_schema if present, otherwise reads fields directly.
    Returns a dict with: altitude, vx, vy, dx_pad, fuel_fraction,
                          sin_theta, cos_theta, omega,
                          leg_contact_left, leg_contact_right
    """
    obs    = state.get("observation", [])
    schema = state.get("observation_schema", [])

    if obs and schema and len(obs) == len(schema):
        return dict(zip(schema, obs))

    # Fallback: read readable fields directly
    pos = state.get("position", {})
    vel = state.get("velocity", {})
    return {
        "altitude":          pos.get("altitude", state.get("altitude", 0.0)),
        "vx":                vel.get("vx",       state.get("vx",       0.0)),
        "vy":                vel.get("vy",       state.get("vy",       0.0)),
        "dx_pad":            state.get("dx_pad",  0.0),
        "fuel_fraction":     state.get("fuel",    1.0),
        "sin_theta":         math.sin(state.get("tilt", 0.0)),
        "cos_theta":         math.cos(state.get("tilt", 0.0)),
        "omega":             state.get("omega",   state.get("angular_velocity", 0.0)),
        "leg_contact_left":  state.get("leg_contact_left",  0),
        "leg_contact_right": state.get("leg_contact_right", 0),
    }


# PD gains — tuned for noisy lander physics
KP_X     =  0.30   # horizontal position error → rotation
KD_X     =  0.50   # horizontal velocity correction
KP_Y     =  1.20   # altitude error → thrust
KD_Y     =  0.80   # vertical velocity correction
KP_TILT  =  1.80   # tilt → stabilize threshold
KD_TILT  =  0.60   # angular velocity threshold

# Thresholds
ALT_FINAL      = 0.08   # below this: switch to gentle landing mode
DX_THRESHOLD   = 0.12   # tolerable horizontal offset
VX_THRESHOLD   = 0.15   # tolerable horizontal speed
VY_THRESHOLD   = -0.35  # max safe descent speed
TILT_THRESHOLD = 0.20   # max tilt (radians) before rotating
OMEGA_THRESHOLD= 0.18   # max angular velocity before stabilizing


def choose_action(obs):
    """
    PD controller returning one of the 7 valid actions.

    Logic layers (highest priority first):
      1. Tilt/spin too large → stabilize
      2. Near ground → idle (let gravity land gently)
      3. Horizontal error → rotate to correct
      4. Need altitude → main thrust
      5. Default → idle
    """
    alt   = obs["altitude"]
    vx    = obs["vx"]
    vy    = obs["vy"]
    dx    = obs["dx_pad"]        # positive = pad is to the right
    theta = math.atan2(obs["sin_theta"], obs["cos_theta"])  # tilt in radians
    omega = obs["omega"]

    # 1. Excessive tilt or spin → stabilize
    if abs(theta) > TILT_THRESHOLD or abs(omega) > OMEGA_THRESHOLD:
        return "stabilize"

    # 2. Very low altitude → cut thrust, let it land
    if alt < ALT_FINAL:
        if abs(vx) > VX_THRESHOLD:
            return "left" if vx > 0 else "right"
        return "idle"

    # 3. Horizontal error — combine with thrust if also need to climb
    need_thrust = (vy < VY_THRESHOLD) or (alt < 0.25 and vy < 0.0)
    dx_error    = KP_X * dx + KD_X * vx   # positive → move right

    if abs(dx_error) > DX_THRESHOLD:
        if dx_error > 0:   # need to move right
            return "main_right" if need_thrust else "right"
        else:              # need to move left
            return "main_left"  if need_thrust else "left"

    # 4. Vertical error — need to climb or slow descent
    if need_thrust:
        return "main"

    # 5. Coast
    return "idle"


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

        obs = extract_obs(state)

        # Emit full state for canvas renderer + telemetry strip
        tprint(f"STATE_JSON:{json.dumps({'position': {'x': state.get('position', {}).get('x', 50), 'altitude': obs['altitude']}, 'velocity': {'vx': obs['vx'], 'vy': obs['vy']}, 'tilt': math.atan2(obs['sin_theta'], obs['cos_theta']), 'fuel': obs['fuel_fraction'], 'landing_pad': state.get('landing_pad', {'x1': 40, 'x2': 60}), 'world_bounds': state.get('world_bounds', {'width': 100, 'height': 80})})}")

        # Log human-readable telemetry
        tprint(f"step={step} alt={obs['altitude']:.3f} vx={obs['vx']:+.3f} vy={obs['vy']:+.3f} fuel={obs['fuel_fraction']:.2f}")

        # Manual override takes priority over controller
        manual_cmd = pop_manual_command()
        if manual_cmd:
            action = manual_cmd
            tprint(f"[MANUAL] → {action}")
        else:
            action = choose_action(obs)
            tprint(f"[AUTO]   → {action}")

        act_resp = safe_call(act, session_id, action)
        status = act_resp.get("status", "continue")

        # Emit updated state from act response
        if "state" in act_resp:
            updated = act_resp["state"]
            updated_obs = extract_obs(updated)
            tprint(f"STATE_JSON:{json.dumps({'position': {'x': updated.get('position', {}).get('x', 50), 'altitude': updated_obs['altitude']}, 'velocity': {'vx': updated_obs['vx'], 'vy': updated_obs['vy']}, 'tilt': math.atan2(updated_obs['sin_theta'], updated_obs['cos_theta']), 'fuel': updated_obs['fuel_fraction'], 'landing_pad': updated.get('landing_pad', state.get('landing_pad', {'x1': 40, 'x2': 60})), 'world_bounds': updated.get('world_bounds', state.get('world_bounds', {'width': 100, 'height': 80}))})}")

        step += 1

    tprint(f"Result: {status.upper()}")
    return status


# ---------------------------------------------------------------------------
# Infinite runner (called by main.py thread)
# ---------------------------------------------------------------------------

def run():
    results = {"win": 0, "lose": 0, "other": 0}
    game_num = 0

    # Clean up any stale command file
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
        net = results["win"] * 10 - results["lose"] * 2
        tprint(f"Totals — W:{results['win']} L:{results['lose']} | Score: {net:+d}")

        time.sleep(1)


if __name__ == "__main__":
    run()