"""
modules/games/tic_tac_toe.py — Tic-Tac-Toe agent.

Runs as an infinite loop: win/tie/lose → new game → repeat.
Thread-safe: uses only local state, no globals.

Decision priority each turn:
  1. Win immediately
  2. Block opponent's immediate win
  3. Create a fork (2 simultaneous threats)
  4. Block opponent's fork
  5. Corner-trap strategy (image: start corner → extend row → trap)
  6. Minimax fallback
"""
import json
import time
import threading
from ..endpoints import newGame, getState, act, stopGame

GAME_ID = 1
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    """Thread-safe print prefixed with thread name."""
    name = threading.current_thread().name
    with _print_lock:
        print(f"[{name}]", *args, **kwargs)

# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def empty_board():
    return {f"{r}{c}": "" for r in range(3) for c in range(3)}


def extract_board(response):
    state = response.get("state", response)
    for key in ("board", "grid", "cells"):
        if key in state:
            raw = state[key]
            if isinstance(raw, list):
                return {f"{r}{c}": (raw[r][c] or "") for r in range(3) for c in range(3)}
            elif isinstance(raw, dict):
                return {k: (v or "") for k, v in raw.items()}
    board = {k: (v or "") for k, v in state.items()
             if len(k) == 2 and k[0].isdigit() and k[1].isdigit()}
    return board if board else None


def board_str(board):
    sym = lambda v: v if v in ("X", "O") else "."
    rows = []
    for r in range(3):
        rows.append(" | ".join(sym(board.get(f"{r}{c}", "")) for c in range(3)))
    return "\n".join(rows)


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


def safe_act(session_id, move, retries=10):
    return safe_call(act, session_id, move, retries=retries)


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------

LINES = [
    ["00","01","02"], ["10","11","12"], ["20","21","22"],
    ["00","10","20"], ["01","11","21"], ["02","12","22"],
    ["00","11","22"], ["02","11","20"],
]
CORNERS = ["00", "02", "20", "22"]


def check_winner(board, player):
    return any(all(board.get(p) == player for p in line) for line in LINES)


def available(board):
    return [k for k, v in board.items() if v == ""]


def find_winning_move(board, player):
    for m in available(board):
        board[m] = player
        won = check_winner(board, player)
        board[m] = ""
        if won:
            return m
    return None


def count_threats(board, player):
    threats = 0
    for line in LINES:
        vals = [board.get(p, "") for p in line]
        if vals.count(player) == 2 and vals.count("") == 1:
            threats += 1
    return threats


def minimax(board, is_maximizing, alpha, beta, depth=0):
    if check_winner(board, "X"): return 10 - depth
    if check_winner(board, "O"): return depth - 10
    moves = available(board)
    if not moves: return 0
    if is_maximizing:
        best = -float("inf")
        for m in moves:
            board[m] = "X"
            best = max(best, minimax(board, False, alpha, beta, depth+1))
            board[m] = ""
            alpha = max(alpha, best)
            if beta <= alpha: break
        return best
    else:
        best = float("inf")
        for m in moves:
            board[m] = "O"
            best = min(best, minimax(board, True, alpha, beta, depth+1))
            board[m] = ""
            beta = min(beta, best)
            if beta <= alpha: break
        return best


def choose_move(board, our_moves):
    avail = available(board)

    # 1. Win immediately
    m = find_winning_move(board, "X")
    if m: return m

    # 2. Block opponent's immediate win
    m = find_winning_move(board, "O")
    if m: return m

    # 3. Create a fork
    for m in avail:
        board[m] = "X"
        threats = count_threats(board, "X")
        board[m] = ""
        if threats >= 2:
            return m

    # 4. Block opponent's fork
    fork_threats = []
    for m in avail:
        board[m] = "O"
        threats = count_threats(board, "O")
        board[m] = ""
        if threats >= 2:
            fork_threats.append(m)

    if len(fork_threats) == 1:
        return fork_threats[0]
    elif len(fork_threats) > 1:
        for m in avail:
            board[m] = "X"
            threats = count_threats(board, "X")
            board[m] = ""
            if threats >= 1 and m not in fork_threats:
                return m

    # 5. Corner-trap strategy (from image)
    turn = len(our_moves)
    if turn == 0:
        return "20"
    if turn == 1:
        if board.get("11") == "O":
            for c in ["00", "02", "22"]:
                if c in avail: return c
        else:
            if "22" in avail: return "22"
    if turn == 2:
        if "02" in avail: return "02"

    # 6. Minimax fallback
    best_s, chosen = -float("inf"), None
    for m in avail:
        board[m] = "X"
        s = minimax(board, False, -float("inf"), float("inf"))
        board[m] = ""
        if s > best_s:
            best_s, chosen = s, m
    return chosen


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

async def play_game_async():
    """Async-compatible wrapper — runs synchronously but is awaitable."""
    return play_game()


def play_game():
    tprint("=== New game ===")
    new_resp = safe_call(newGame, GAME_ID)
    session_id = new_resp["gamesessionid"]
    tprint(f"Session {session_id}")

    board = empty_board()
    status = "continue"
    our_moves = []

    while status == "continue":
        tprint("\n" + board_str(board))
        move = choose_move(board, our_moves)
        if move is None:
            break

        tprint(f"→ {move}")
        our_moves.append(move)

        resp = safe_act(session_id, move)
        status = resp.get("status", "continue")

        updated = extract_board(resp)
        if updated:
            board = updated
        else:
            board[move] = "X"

        # Emit full board state for dashboard rendering
        board_2d = [[board.get(f"{r}{c}", "") for c in range(3)] for r in range(3)]
        tprint(f"STATE_JSON:{json.dumps({'board': board_2d, 'current_player': 'X', 'status': status})}")

    tprint("\n" + board_str(board))
    tprint(f"Result: {status.upper()}")
    return status


# ---------------------------------------------------------------------------
# Infinite runner (called by main.py thread)
# ---------------------------------------------------------------------------

def run():
    """
    Infinite loop: play a game, log result, repeat.
    Designed to run inside a dedicated thread.
    """
    results = {"win": 0, "lose": 0, "tie": 0, "other": 0}
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
        net = results["win"] + results["tie"] - results["lose"]
        tprint(f"Totals — W:{results['win']} L:{results['lose']} T:{results['tie']} | Score: {net:+d}")

        time.sleep(1)  # gap between games


if __name__ == "__main__":
    run()