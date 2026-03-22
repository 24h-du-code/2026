"""
modules/games/adaptive_traffic_racing.py — Adaptive Traffic Racing agent (Game #5)

Claude API makes the driving decision every step, with an emergency fallback
heuristic if Claude is too slow or unavailable.
Runs as an infinite loop: win/lose → new game → repeat.
Thread-safe: uses only local state, no globals.
"""
import json
import time
import threading
import requests
from ..endpoints import newGame, getState, act, stopGame

GAME_ID        = 5
OLLAMA_TIMEOUT = 10.0
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen2.5:3b"
DANGER_GAP     = 3.0
SAFE_GAP       = 7.0
MAX_SPEED      = 5.0
WARN_GAP       = 8.0
HARD_DANGER    = 2.0
VALID_ACTIONS  = {"left", "right", "keep", "accelerate", "brake"}

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
# State formatter for Claude prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert racing car AI agent competing in a 3-lane traffic race.
Your goal: reach the finish line as fast as possible without collisions.

## Game rules
- 3 lanes: 0 (left), 1 (center), 2 (right)
- Actions: left, right, keep, accelerate, brake
- Win (+6-7 pts) by completing track_length. Lose (-2 pts) on collision.

## CRITICAL: You can only move ONE lane per step.
If you are in lane 2 and lanes 1 AND 2 are both blocked ahead, you CANNOT reach lane 0 in time.
Start escaping 2-3 steps EARLY — before both adjacent lanes close.
Never get boxed in: always keep at least one safe adjacent lane available.

## Collision detection
- relative_speed < 0 on a vehicle AHEAD = you are CLOSING on them = DANGER
- The closer + faster the closing speed, the more urgent
- safe_now=false = imminent collision in that lane, never enter it
- ahead < 3.0 = dangerous even if safe_now=true

## All-slower-traffic pattern (hidden style)
- If ALL nearby vehicles have relative_speed < 0, all traffic is slower than you
- You will close on every car ahead — brake and create distance
- Do NOT accelerate when all traffic is slower. Match their speed.

## Two-step escape rule
- Before choosing a lane, ask: from lane X, can I still escape in 1 more step?
- Lane 1 (center) has 2 escape directions — prefer it when choices are equal

Respond with EXACTLY one word: the action name."""


def format_state(state, history):
    lane  = state['lane'];  speed = state['speed']
    prog  = state['progress']; tlen = state['track_length']
    step  = state['step'];  maxs = state['max_steps']
    pct   = prog / tlen * 100
    pace  = prog / max(step, 1); needed = tlen / maxs
    pace_note = "BEHIND — push harder" if pace < needed * 0.9 else "on pace"

    lines = [
        f"Step {step}/{maxs} | {prog:.1f}/{tlen:.0f} ({pct:.0f}%) | "
        f"Speed {speed:.1f} | Lane {lane} | {pace_note}", "", "GAPS:"
    ]
    for g in sorted(state['lane_gaps'], key=lambda x: x['lane']):
        you = " ←YOU" if g['lane'] == lane else ""
        lines.append(f"  L{g['lane']}: ahead={g['ahead']:.1f} "
                     f"safe={'Y' if g['safe_now'] else '⚠NO'}{you}")
    lines += ["", "NEARBY (+ = ahead of you):"]
    for v in sorted(state['nearby_vehicles'],
                    key=lambda v: abs(v['distance']) if v['distance'] is not None else 999)[:6]:
        warn = ""
        if (v['distance'] or 0) > 0 and v['relative_speed'] < -0.15: warn = " ⚠CLOSING"
        if (v['distance'] or 0) < 0 and v['relative_speed'] > 0.2:   warn = " ⚠CATCHING YOU"
        lines.append(f"  L{v['lane']} d={(v['distance'] or 0):+.1f} rs={v['relative_speed']:+.3f}{warn}")

    occ_lines = []
    for ld in state.get('sensor_window', []):
        occ = [(c['offset'], c['relative_speed']) for c in ld['cells'] if c['occupied']]
        if occ:
            occ_lines.append(f"  L{ld['lane']}: " +
                "  ".join(f"@{o:+d}(rs={rs:+.3f})" for o, rs in occ))
    if occ_lines:
        lines += ["", "SENSOR:"] + occ_lines

    if history:
        lines += ["", "RECENT: " + " → ".join(
            f"{a}({'ok' if s == 'continue' else s})" for a, s in history[-5:])]

    lines += ["", "Action?"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emergency heuristic fallback (used if Claude is unavailable/too slow)
# ---------------------------------------------------------------------------

def emergency_fallback(state):
    lane  = state['lane']; speed = state['speed']
    gaps  = state['lane_gaps']; nearby = state['nearby_vehicles']

    def ginfo(l):
        return next((g for g in gaps if g['lane'] == l), None)

    cur    = ginfo(lane)
    ahead  = (cur['ahead'] or 999.0) if cur else 999.0
    safe_now = cur['safe_now'] if cur else True

    ahead_veh = [v for v in nearby if v['lane'] == lane and (v['distance'] or 0) > 0]
    closing = 0.0
    if ahead_veh:
        c = min(ahead_veh, key=lambda v: v['distance'] or 999)
        if (c['relative_speed'] or 0) < 0:
            closing = abs(c['relative_speed'] or 0)
    ttc = ahead / (closing + 0.01) if closing > 0 else 999.0

    fwd_veh = [v for v in nearby if (v['distance'] or 0) > 0]
    all_closing = (len(fwd_veh) >= 3 and
                   all((v['relative_speed'] or 0) < -0.05 for v in fwd_veh) and
                   sum(1 for v in fwd_veh if (v['relative_speed'] or 0) < -0.3) >= 2)

    def escape_score(tl):
        g = ginfo(tl)
        if g is None or not g['safe_now']: return -1
        return sum(1 for nl in [tl-1, tl, tl+1]
                   if 0 <= nl <= 2 and ginfo(nl) and ginfo(nl)['safe_now']
                   and (ginfo(nl)['ahead'] or 0) > HARD_DANGER)

    immediate = (not safe_now) or (ahead < DANGER_GAP) or (ttc < 2.5) or (ahead < 6.0 and closing > 0.3)
    if immediate:
        best_adj, best_s = None, -1
        for adj in [lane-1, lane+1]:
            if 0 <= adj <= 2:
                g = ginfo(adj)
                if g and g['safe_now']:
                    score = (g['ahead'] or 0) + escape_score(adj) * 3
                    if score > best_s: best_s = score; best_adj = adj
        if best_adj is not None:
            return 'left' if best_adj < lane else 'right'
        return 'brake'

    if ahead < WARN_GAP and closing > 0.2:
        for adj in [lane-1, lane+1]:
            if 0 <= adj <= 2:
                g = ginfo(adj)
                if g and g['safe_now'] and (g['ahead'] or 0) > ahead + 1.5:
                    return 'left' if adj < lane else 'right'

    if not all_closing:
        best_lane, best_score = lane, ahead + escape_score(lane) * 5
        for adj in [lane-1, lane+1]:
            if 0 <= adj <= 2:
                g = ginfo(adj)
                if g and g['safe_now']:
                    s = (g['ahead'] or 0) + escape_score(adj) * 5 - 2
                    if s > best_score: best_score = s; best_lane = adj
        if best_lane != lane:
            return 'left' if best_lane < lane else 'right'

    if all_closing and ahead < 15 and speed > 1.0:
        max_cl = max((abs(v['relative_speed'] or 0) for v in fwd_veh), default=0)
        return 'brake' if max_cl >= 0.3 else 'keep'

    if ahead > SAFE_GAP and speed < MAX_SPEED: return 'accelerate'
    if ahead < DANGER_GAP * 1.5 or (ahead < 8.0 and closing > 0.15): return 'brake'
    return 'keep'


def is_safe(action, state):
    lane = state['lane']; gaps = state['lane_gaps']
    tl = lane + (-1 if action == 'left' else 1 if action == 'right' else 0)
    if tl < 0 or tl > 2: return False
    g = next((g for g in gaps if g['lane'] == tl), None)
    if not g: return True
    return g['safe_now'] and (g['ahead'] or 0) >= HARD_DANGER


def ollama_decide(state, history):
    try:
        prompt = f"{SYSTEM_PROMPT}\n\n{format_state(state, history)}"
        r = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 10, "temperature": 0.0}
            },
            timeout=OLLAMA_TIMEOUT
        )
        r.raise_for_status()
        raw = r.json()["response"].strip().lower()
        for w in VALID_ACTIONS:
            if w in raw:
                return w if is_safe(w, state) else emergency_fallback(state)
    except Exception as e:
        tprint(f"[Ollama] {e} — using fallback")
    return emergency_fallback(state)


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def play_game():
    tprint("=== New game ===")
    new_resp = safe_call(newGame, GAME_ID)
    session_id = new_resp["gamesessionid"]
    tprint(f"Session {session_id}")

    status  = "continue"
    history = []

    while status == "continue":
        state_resp = safe_call(getState, session_id)
        state = state_resp.get("state", state_resp)

        if state.get("done"):
            break

        action = ollama_decide(state, history)
        tprint(f"lane={state['lane']} speed={state['speed']:.1f} → {action}")

        # Emit full state for dashboard — all fields needed by renderAdaptiveTrafficRacing
        tprint(f"STATE_JSON:{json.dumps({'lane': state.get('lane', 1), 'speed': state.get('speed', 1.0), 'progress': state.get('progress', 0), 'track_length': state.get('track_length', 100), 'step': state.get('step', 0), 'max_steps': state.get('max_steps', 200), 'nearby_vehicles': state.get('nearby_vehicles', []), 'lane_gaps': state.get('lane_gaps', []), 'sensor_window': state.get('sensor_window', []), 'last_action': action})}")

        act_resp = safe_call(act, session_id, action)
        status = act_resp.get("status", "continue")
        history.append((action, status))

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
        net = results["win"] * 6 - results["lose"] * 2
        tprint(f"Totals — W:{results['win']} L:{results['lose']} | Score: {net:+d}")

        time.sleep(1)


if __name__ == "__main__":
    run()