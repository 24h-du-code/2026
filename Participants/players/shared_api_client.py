import atexit
import requests
import time
import logging
import os

# Configuration constants
DEFAULT_MAX_CALLS_PER_SECOND = float(os.getenv("PLAYER_MAX_CALLS_PER_SECOND", "1"))
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("PLAYER_REQUEST_TIMEOUT_SECONDS", "10"))


class GameAPIClient:
    TERMINAL_STATUSES = {"win", "lose", "tie", "max_steps"}

    def __init__(
        self,
        server_url,
        token,
        *,
        max_calls_per_second=None,
        request_timeout=None,
        cleanup_on_exit=True,
    ):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.auth_mode = "bearer"
        self.max_calls_per_second = (
            float(max_calls_per_second)
            if max_calls_per_second is not None
            else DEFAULT_MAX_CALLS_PER_SECOND
        )
        self.request_timeout = (
            float(request_timeout)
            if request_timeout is not None
            else DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        self.last_call_time = 0
        self._games_by_name = None
        self._active_session_ids = set()
        self._cleanup_registered = False
        if cleanup_on_exit:
            self._register_cleanup()

    def _register_cleanup(self):
        if not self._cleanup_registered:
            atexit.register(self.cleanup_active_sessions)
            self._cleanup_registered = True

    def _track_session(self, session_id):
        if session_id is not None:
            self._active_session_ids.add(int(session_id))

    def _untrack_session(self, session_id):
        if session_id is not None:
            self._active_session_ids.discard(int(session_id))

    def _build_auth_headers(self, auth_mode=None):
        auth_mode = auth_mode or self.auth_mode
        if auth_mode == "raw":
            return {"Authorization": self.token}
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.request_timeout)
        base_headers = dict(kwargs.pop("headers", {}))
        headers = dict(base_headers)
        headers.update(self._build_auth_headers())
        response = requests.request(method, url, headers=headers, **kwargs)

        if response.status_code == 401 and self.auth_mode == "bearer":
            fallback_headers = dict(base_headers)
            fallback_headers.update(self._build_auth_headers("raw"))
            fallback_response = requests.request(method, url, headers=fallback_headers, **kwargs)
            if fallback_response.ok:
                logging.warning("Server rejected Bearer auth, falling back to raw Authorization header")
                self.auth_mode = "raw"
                return fallback_response

        return response

    def _enforce_rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 1.0 / self.max_calls_per_second
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        self.last_call_time = time.time()

    def list_games(self):
        self._enforce_rate_limit()
        url = f"{self.server_url}/api/list_games/"
        response = self._request("get", url)
        response.raise_for_status()
        games = response.json()
        self._games_by_name = {game["name"]: game for game in games}
        return games

    def get_game_id_by_name(self, game_name, *, refresh=False):
        if refresh or self._games_by_name is None:
            self.list_games()
        game = self._games_by_name.get(game_name) if self._games_by_name else None
        return game["id"] if game else None

    def start_game(self, game_id):
        self._enforce_rate_limit()
        url = f"{self.server_url}/api/newgame/"
        response = self._request("post", url, json={"idgame": game_id})
        if response.status_code == 409:
            error_data = response.json()
            existing_id = error_data.get("existing_session_id")
            if existing_id:
                logging.warning(
                    "Active session %s already exists for game %s, reusing it",
                    existing_id,
                    game_id,
                )
                self._track_session(existing_id)
                return {
                    "gamesessionid": existing_id,
                    "reused_existing_session": True,
                    "action_list": None,
                }
            raise RuntimeError("Conflict but no existing session id")
        response.raise_for_status()
        payload = response.json()
        self._track_session(payload.get("gamesessionid"))
        return payload

    def get_state(self, session_id):
        self._enforce_rate_limit()
        url = f"{self.server_url}/api/get_state/"
        response = self._request("get", url, params={"gamesessionid": session_id})
        response.raise_for_status()
        return response.json()

    def act(self, session_id, action):
        self._enforce_rate_limit()
        url = f"{self.server_url}/api/act/"
        response = self._request(
            "post",
            url,
            json={"gamesessionid": session_id, "action": action}
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in self.TERMINAL_STATUSES:
            self._untrack_session(session_id)
        return payload

    def stop_game(self, session_id, *, allow_missing=False):
        self._enforce_rate_limit()
        url = f"{self.server_url}/api/stop_game/"
        response = self._request("post", url, json={"gamesessionid": session_id})
        if allow_missing and response.status_code in {400, 404}:
            self._untrack_session(session_id)
            return response.json()
        response.raise_for_status()
        self._untrack_session(session_id)
        return response.json()

    def cleanup_active_sessions(self):
        for session_id in list(self._active_session_ids):
            try:
                self.stop_game(session_id, allow_missing=True)
            except Exception as exc:
                logging.warning("Failed to stop session %s during cleanup: %s", session_id, exc)
