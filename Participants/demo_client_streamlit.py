"""
Demo QA client for the game server using Streamlit.
"""

from __future__ import annotations

from copy import deepcopy

import requests
import streamlit as st

from players.shared_api_client import (
    DEFAULT_MAX_CALLS_PER_SECOND,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    GameAPIClient,
)

DEFAULT_SERVER_URL = "http://localhost:15001"
AUTO_REFRESH_INTERVALS = ("2s", "5s")
TERMINAL_STATUSES = {"win", "lose", "tie", "max_steps"}
INACTIVE_STATUSES = TERMINAL_STATUSES | {"cancelled", "inactive", "unavailable"}
UNKNOWN_GAME_NAME = "Unknown game"
UNSET = object()


def init_state():
    """Initialize Streamlit session state defaults."""
    defaults = {
        "server_url": DEFAULT_SERVER_URL,
        "token": "",
        "request_timeout": float(DEFAULT_REQUEST_TIMEOUT_SECONDS),
        "max_calls_per_second": float(DEFAULT_MAX_CALLS_PER_SECOND),
        "games": [],
        "selected_game_name": None,
        "known_sessions": {},
        "current_session_id": None,
        "current_game_id": None,
        "current_game_name": None,
        "current_state": None,
        "current_actions": {},
        "current_status": None,
        "current_remaining_steps": None,
        "current_session_reused": False,
        "last_payload": None,
        "message": None,
        "manual_action_enabled": False,
        "manual_action_code": "",
        "selected_action_code": None,
        "attach_session_id_input": "",
        "auto_refresh_enabled": False,
        "auto_refresh_interval": AUTO_REFRESH_INTERVALS[0],
        "api_auth_mode": "bearer",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def set_message(level, text):
    """Store a message to be rendered on the next pass."""
    st.session_state.message = {"level": level, "text": text}


def render_message(container=None):
    """Render the latest stored message."""
    message = st.session_state.message
    if not message:
        return

    level = message.get("level", "info")
    text = message.get("text", "")
    render_target = container if container is not None else st
    renderer = getattr(render_target, level, render_target.info)
    renderer(text)


def build_client():
    """Create the shared API client from the current UI config."""
    server_url = str(st.session_state.server_url).strip()
    token = str(st.session_state.token).strip()

    if not server_url:
        raise ValueError("Please enter a server URL.")
    if not token:
        raise ValueError("Please enter an authentication token.")

    client = GameAPIClient(
        server_url,
        token,
        max_calls_per_second=float(st.session_state.max_calls_per_second),
        request_timeout=float(st.session_state.request_timeout),
        cleanup_on_exit=False,
    )
    client.auth_mode = st.session_state.api_auth_mode
    return client


def persist_client_state(client):
    """Persist mutable client settings between reruns."""
    st.session_state.api_auth_mode = client.auth_mode


def get_api_error_message(exc):
    """Extract a readable error message from API exceptions."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            error_message = payload.get("error") or payload.get("message")
            if error_message:
                return error_message

        text = getattr(response, "text", "")
        if text:
            return text

        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return f"HTTP {status_code}"

    message = str(exc).strip()
    return message or exc.__class__.__name__


def response_status_code(exc):
    """Return the HTTP status code for a requests exception when available."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def find_game_by_name(game_name):
    """Return a loaded game record by name."""
    for game in st.session_state.games:
        if game["name"] == game_name:
            return game
    return None


def get_game_name_for_id(game_id):
    """Return a game name from the loaded catalog."""
    if game_id is None:
        return UNKNOWN_GAME_NAME
    for game in st.session_state.games:
        if game["id"] == game_id:
            return game["name"]
    return UNKNOWN_GAME_NAME


def is_locally_active(status):
    """Return whether the local session record is still actionable."""
    return status not in INACTIVE_STATUSES


def format_status(status):
    """Render a user-friendly status label."""
    mapping = {
        None: "Unknown",
        "active": "Active",
        "continue": "Active",
        "win": "Win",
        "lose": "Lose",
        "tie": "Tie",
        "max_steps": "Max steps",
        "cancelled": "Cancelled",
        "inactive": "Inactive",
        "unavailable": "Unavailable",
    }
    return mapping.get(status, str(status).replace("_", " ").title())


def default_session_record(session_id):
    """Create a default local record for a known session."""
    return {
        "session_id": int(session_id),
        "game_id": None,
        "game_name": UNKNOWN_GAME_NAME,
        "status": "active",
        "remaining_steps": None,
        "reused_existing_session": False,
    }


def get_known_session(session_id):
    """Fetch a known session record from local state."""
    return st.session_state.known_sessions.get(str(int(session_id)))


def remember_session(
    session_id,
    *,
    game_id=UNSET,
    game_name=UNSET,
    status=UNSET,
    remaining_steps=UNSET,
    reused_existing_session=UNSET,
):
    """Insert or update a session record in local state."""
    sessions = dict(st.session_state.known_sessions)
    key = str(int(session_id))
    record = dict(sessions.get(key, default_session_record(session_id)))

    updates = {
        "game_id": game_id,
        "game_name": game_name,
        "status": status,
        "remaining_steps": remaining_steps,
        "reused_existing_session": reused_existing_session,
    }
    for field, value in updates.items():
        if value is not UNSET:
            record[field] = value

    if not record.get("game_name") and record.get("game_id") is not None:
        record["game_name"] = get_game_name_for_id(record["game_id"])
    if not record.get("game_name"):
        record["game_name"] = UNKNOWN_GAME_NAME

    sessions[key] = record
    st.session_state.known_sessions = sessions
    return record


def forget_session(session_id):
    """Remove a session from the local session registry."""
    key = str(int(session_id))
    sessions = dict(st.session_state.known_sessions)
    sessions.pop(key, None)
    st.session_state.known_sessions = sessions

    if st.session_state.current_session_id == int(session_id):
        clear_current_session()

    set_message("info", f"Removed local reference to session {session_id}.")


def clear_current_session():
    """Clear the attached session view without deleting local history."""
    st.session_state.current_session_id = None
    st.session_state.current_game_id = None
    st.session_state.current_game_name = None
    st.session_state.current_state = None
    st.session_state.current_actions = {}
    st.session_state.current_status = None
    st.session_state.current_remaining_steps = None
    st.session_state.current_session_reused = False
    st.session_state.selected_action_code = None


def set_current_session(
    session_id,
    *,
    game_id=UNSET,
    game_name=UNSET,
    state=UNSET,
    actions=UNSET,
    status=UNSET,
    remaining_steps=UNSET,
    reused_existing_session=UNSET,
    payload=UNSET,
):
    """Attach a session to the main session panel."""
    record = remember_session(
        session_id,
        game_id=game_id,
        game_name=game_name,
        status=status,
        remaining_steps=remaining_steps,
        reused_existing_session=reused_existing_session,
    )

    st.session_state.current_session_id = record["session_id"]
    st.session_state.current_game_id = record.get("game_id")
    st.session_state.current_game_name = record.get("game_name")
    st.session_state.current_status = record.get("status")
    st.session_state.current_remaining_steps = record.get("remaining_steps")
    st.session_state.current_session_reused = record.get("reused_existing_session", False)

    if state is not UNSET:
        st.session_state.current_state = state
    if actions is not UNSET:
        st.session_state.current_actions = dict(actions or {})
        sync_selected_action()
    if payload is not UNSET:
        st.session_state.last_payload = payload


def sync_selected_action():
    """Ensure the selected guided action is valid for the current action list."""
    current_actions = dict(st.session_state.current_actions or {})
    if not current_actions:
        st.session_state.selected_action_code = None
        return

    selected_action = st.session_state.selected_action_code
    if selected_action not in current_actions:
        st.session_state.selected_action_code = next(iter(current_actions))


def load_games(*, success_prefix="Loaded"):
    """Load the remote game catalog through the shared client."""
    try:
        client = build_client()
        games = client.list_games()
        persist_client_state(client)
    except Exception as exc:
        set_message("error", f"Could not load games: {get_api_error_message(exc)}")
        return False

    st.session_state.games = games
    if games:
        game_names = [game["name"] for game in games]
        if st.session_state.selected_game_name not in game_names:
            st.session_state.selected_game_name = game_names[0]
    else:
        st.session_state.selected_game_name = None

    set_message("success", f"{success_prefix} {len(games)} games from the server.")
    return True


def hydrate_session(
    session_id,
    *,
    announce=None,
    game_id=UNSET,
    game_name=UNSET,
    status=UNSET,
    reused_existing_session=UNSET,
    show_message=True,
):
    """Fetch and attach the latest visible state for a session."""
    try:
        client = build_client()
        payload = client.get_state(session_id)
        persist_client_state(client)
    except Exception as exc:
        status_code = response_status_code(exc)
        if status_code == 404:
            remember_session(session_id, status="unavailable")
            if st.session_state.current_session_id == int(session_id):
                set_current_session(
                    session_id,
                    status="unavailable",
                    actions={},
                    game_id=game_id,
                    game_name=game_name,
                )
        if show_message:
            set_message("error", f"Could not load session {session_id}: {get_api_error_message(exc)}")
        return False

    existing_record = get_known_session(session_id) or default_session_record(session_id)
    local_status = existing_record.get("status", "active") if status is UNSET else status
    actions = payload.get("action_list") or {}
    if not is_locally_active(local_status):
        actions = {}

    set_current_session(
        session_id,
        game_id=existing_record.get("game_id") if game_id is UNSET else game_id,
        game_name=existing_record.get("game_name") if game_name is UNSET else game_name,
        state=payload.get("state"),
        actions=actions,
        status=local_status,
        remaining_steps=existing_record.get("remaining_steps"),
        reused_existing_session=(
            existing_record.get("reused_existing_session", False)
            if reused_existing_session is UNSET
            else reused_existing_session
        ),
        payload=payload,
    )

    if show_message and announce:
        set_message("success", announce)
    return True


def start_selected_game():
    """Start or resume a session for the currently selected game."""
    selected_game = find_game_by_name(st.session_state.selected_game_name)
    if not selected_game:
        set_message("warning", "Please load the game catalog and choose a game first.")
        return

    try:
        client = build_client()
        payload = client.start_game(selected_game["id"])
        persist_client_state(client)
    except Exception as exc:
        set_message("error", f"Could not start the game: {get_api_error_message(exc)}")
        return

    session_id = int(payload["gamesessionid"])
    reused_existing_session = bool(payload.get("reused_existing_session"))
    start_message = (
        f"Resumed existing session {session_id} for {selected_game['name']}."
        if reused_existing_session
        else f"Started session {session_id} for {selected_game['name']}."
    )

    set_current_session(
        session_id,
        game_id=selected_game["id"],
        game_name=selected_game["name"],
        state=st.session_state.current_state if st.session_state.current_session_id == session_id else None,
        actions=payload.get("action_list") or {},
        status="active",
        remaining_steps=None,
        reused_existing_session=reused_existing_session,
        payload=payload,
    )

    hydrated = hydrate_session(
        session_id,
        announce=start_message,
        game_id=selected_game["id"],
        game_name=selected_game["name"],
        status="active",
        reused_existing_session=reused_existing_session,
        show_message=True,
    )
    if not hydrated:
        set_message(
            "warning",
            f"Session {session_id} is attached locally, but the initial state could not be loaded.",
        )


def refresh_current_session(*, show_message=True, source="manual"):
    """Refresh the attached session state."""
    session_id = st.session_state.current_session_id
    if session_id is None:
        if show_message:
            set_message("warning", "No session is currently attached.")
        return False

    announce = None
    if show_message:
        announce = (
            f"Session {session_id} refreshed."
            if source == "manual"
            else f"Session {session_id} auto-refreshed."
        )

    return hydrate_session(
        session_id,
        announce=announce,
        game_id=st.session_state.current_game_id,
        game_name=st.session_state.current_game_name,
        status=st.session_state.current_status or "active",
        reused_existing_session=st.session_state.current_session_reused,
        show_message=show_message,
    )


def current_action_value():
    """Resolve the action code to submit from guided or manual mode."""
    if st.session_state.manual_action_enabled:
        return str(st.session_state.manual_action_code).strip()
    return st.session_state.selected_action_code


def submit_current_action():
    """Submit an action for the attached session."""
    session_id = st.session_state.current_session_id
    if session_id is None:
        set_message("warning", "No session is currently attached.")
        return

    if not is_locally_active(st.session_state.current_status):
        set_message("warning", "The attached session is no longer locally marked active.")
        return

    action = current_action_value()
    if not action:
        set_message("warning", "Please choose or enter an action first.")
        return

    try:
        client = build_client()
        payload = client.act(session_id, action)
        persist_client_state(client)
    except Exception as exc:
        status_code = response_status_code(exc)
        if status_code == 404:
            remember_session(session_id, status="unavailable")
            st.session_state.current_status = "unavailable"
            st.session_state.current_actions = {}
        set_message("error", f"Could not submit the action: {get_api_error_message(exc)}")
        return

    status = payload.get("status", "active")
    actions = payload.get("action_list") or {}
    remaining_steps = payload.get("remaining_steps")
    if status in TERMINAL_STATUSES:
        actions = {}

    set_current_session(
        session_id,
        game_id=st.session_state.current_game_id,
        game_name=st.session_state.current_game_name,
        state=payload.get("state"),
        actions=actions,
        status=status,
        remaining_steps=remaining_steps,
        reused_existing_session=False,
        payload=payload,
    )

    if status in TERMINAL_STATUSES:
        set_message("success", f"Game ended with status: {format_status(status)}.")
        if status == "win":
            st.balloons()
    else:
        set_message(
            "info",
            f"Action applied. Status: {format_status(status)}. Remaining steps: {remaining_steps}.",
        )


def stop_session(session_id, *, message_prefix="Stopped"):
    """Stop a locally known session if it is still active."""
    try:
        client = build_client()
        payload = client.stop_game(session_id)
        persist_client_state(client)
    except Exception as exc:
        status_code = response_status_code(exc)
        if status_code == 404:
            remember_session(session_id, status="unavailable", remaining_steps=None)
            if st.session_state.current_session_id == int(session_id):
                st.session_state.current_status = "unavailable"
                st.session_state.current_actions = {}
        elif status_code == 400:
            remember_session(session_id, status="inactive")
            if st.session_state.current_session_id == int(session_id):
                st.session_state.current_status = "inactive"
                st.session_state.current_actions = {}
        set_message("error", f"Could not stop session {session_id}: {get_api_error_message(exc)}")
        return False

    remember_session(session_id, status="cancelled", remaining_steps=None)
    if st.session_state.current_session_id == int(session_id):
        st.session_state.current_status = "cancelled"
        st.session_state.current_actions = {}
        st.session_state.last_payload = payload
    set_message("success", f"{message_prefix} session {session_id}.")
    return True


def attach_session_by_id():
    """Attach a session by a manually entered session ID."""
    raw_value = str(st.session_state.attach_session_id_input).strip()
    if not raw_value:
        set_message("warning", "Please enter a session ID to attach.")
        return

    try:
        session_id = int(raw_value)
    except ValueError:
        set_message("error", "Session IDs must be integers.")
        return

    known_record = get_known_session(session_id) or {}
    announce = f"Attached session {session_id}."
    if known_record.get("status") == "unavailable":
        announce = f"Reopened local record for session {session_id}."

    hydrated = hydrate_session(
        session_id,
        announce=announce,
        game_id=known_record.get("game_id", UNSET),
        game_name=known_record.get("game_name", UNSET),
        status=known_record.get("status", "active"),
        reused_existing_session=known_record.get("reused_existing_session", False),
        show_message=True,
    )
    if hydrated:
        return


def render_sidebar():
    """Render the sidebar configuration and utilities."""
    with st.sidebar:
        st.header("Connection")
        st.text_input("Server URL", key="server_url")
        st.text_input("Team token", key="token", type="password")
        st.number_input(
            "Request timeout (seconds)",
            min_value=1.0,
            step=1.0,
            key="request_timeout",
        )
        st.number_input(
            "Max calls per second",
            min_value=0.1,
            step=0.5,
            key="max_calls_per_second",
        )

        if st.button("Test connection", key="test_connection"):
            load_games(success_prefix="Connection successful. Loaded")

        if st.button("Refresh games", key="refresh_games"):
            load_games(success_prefix="Reloaded")

        st.divider()
        st.subheader("Attach an existing session")
        st.text_input("Session ID to attach", key="attach_session_id_input")
        if st.button("Attach session by ID", key="attach_session_button"):
            attach_session_by_id()

        st.divider()
        st.subheader("Auto-refresh")
        st.checkbox("Enable auto-refresh", key="auto_refresh_enabled")
        st.selectbox("Auto-refresh interval", AUTO_REFRESH_INTERVALS, key="auto_refresh_interval")
        st.caption("Auto-refresh only runs while the attached session is locally marked active.")


def render_game_catalog():
    """Render the remote game catalog and start controls."""
    st.header("Available Games")
    games = list(st.session_state.games)
    if not games:
        st.info("Use the sidebar to test the connection and load the game catalog.")
        return

    st.caption(f"{len(games)} games loaded from the server.")
    game_names = [game["name"] for game in games]
    st.selectbox("Select a game", game_names, key="selected_game_name")
    selected_game = find_game_by_name(st.session_state.selected_game_name)

    if selected_game:
        st.write(f"Selected game ID: `{selected_game['id']}`")
        with st.expander("Selected game payload", expanded=False):
            st.json(selected_game)

    if st.button("Start or resume selected game", key="start_game_button"):
        start_selected_game()


def render_auto_refresh_fragment():
    """Render and schedule best-effort auto-refresh using Streamlit fragments."""
    run_every = (
        st.session_state.auto_refresh_interval
        if st.session_state.auto_refresh_enabled
        and st.session_state.current_session_id is not None
        and is_locally_active(st.session_state.current_status)
        else None
    )

    @st.fragment(run_every=run_every)
    def auto_refresh_panel():
        if run_every:
            refresh_current_session(show_message=False, source="auto")
            st.caption(
                f"Auto-refresh is active every {run_every} while the session stays locally active."
            )
        elif st.session_state.auto_refresh_enabled and st.session_state.current_session_id is not None:
            st.caption("Auto-refresh is paused because this session is no longer locally active.")
        else:
            st.caption("Auto-refresh is off.")

    auto_refresh_panel()


def render_current_session():
    """Render the current session panel."""
    st.header("Current Session")
    session_id = st.session_state.current_session_id
    if session_id is None:
        st.info("No session is attached yet.")
        return

    cols = st.columns(5)
    cols[0].metric("Session ID", str(session_id))
    cols[1].metric("Game", st.session_state.current_game_name or UNKNOWN_GAME_NAME)
    cols[2].metric("Status", format_status(st.session_state.current_status))
    remaining_steps = st.session_state.current_remaining_steps
    cols[3].metric(
        "Remaining steps",
        str(remaining_steps) if remaining_steps is not None else "Unknown",
    )
    cols[4].metric(
        "Session source",
        "Reused" if st.session_state.current_session_reused else "New/attached",
    )

    if st.session_state.current_session_reused:
        st.info("This session was reused because an active session already existed on the server.")

    action_cols = st.columns(2)
    if action_cols[0].button("Refresh attached session", key="refresh_session_button"):
        refresh_current_session(show_message=True, source="manual")

    if is_locally_active(st.session_state.current_status):
        if action_cols[1].button("Stop attached session", key="stop_session_button"):
            stop_session(session_id)
            st.rerun()

    render_auto_refresh_fragment()

    st.subheader("Current state")
    if st.session_state.current_state is None:
        st.warning("No state payload is available yet for this session.")
    else:
        st.json(st.session_state.current_state)

    st.subheader("Actions")
    current_actions = dict(st.session_state.current_actions or {})
    if current_actions:
        sync_selected_action()
        action_codes = list(current_actions.keys())
        selected_action_code = st.selectbox(
            "Available action",
            action_codes,
            index=action_codes.index(st.session_state.selected_action_code),
        )
        st.session_state.selected_action_code = selected_action_code
        action_description = current_actions.get(selected_action_code)
        if action_description:
            st.caption(action_description)
    else:
        st.info("No guided actions are currently available for this session.")

    st.toggle("Advanced manual action mode", key="manual_action_enabled")
    if st.session_state.manual_action_enabled:
        st.text_input("Manual action code", key="manual_action_code")

    if st.button(
        "Submit action",
        key="submit_action_button",
        disabled=not is_locally_active(st.session_state.current_status),
    ):
        submit_current_action()
        st.rerun()

    st.subheader("Available actions payload")
    st.json(current_actions)

    with st.expander("Raw payload", expanded=False):
        st.json(st.session_state.last_payload or {})


def render_known_sessions():
    """Render locally known sessions and local QA actions."""
    st.header("Known Sessions")
    sessions = list(st.session_state.known_sessions.values())
    if not sessions:
        st.info("No local session history yet.")
        return

    st.caption("These session records live only in the browser session for quick QA flows.")
    for record in sessions:
        session_id = record["session_id"]
        left, middle, right = st.columns([3, 2, 3])
        label = f"Session `{session_id}` - {record.get('game_name') or UNKNOWN_GAME_NAME}"
        left.markdown(f"**{label}**")
        left.caption(
            f"Status: {format_status(record.get('status'))} | "
            f"Remaining steps: {record.get('remaining_steps') if record.get('remaining_steps') is not None else 'Unknown'}"
        )

        if middle.button("Open", key=f"open_known_{session_id}"):
            hydrate_session(
                session_id,
                announce=f"Attached session {session_id}.",
                game_id=record.get("game_id", UNSET),
                game_name=record.get("game_name", UNSET),
                status=record.get("status", "active"),
                reused_existing_session=record.get("reused_existing_session", False),
                show_message=True,
            )
            st.rerun()

        if is_locally_active(record.get("status")):
            if middle.button("Stop", key=f"stop_known_{session_id}"):
                stop_session(session_id, message_prefix="Stopped")
                st.rerun()

        if right.button("Forget locally", key=f"forget_known_{session_id}"):
            forget_session(session_id)
            st.rerun()


def main():
    """Render the Streamlit demo QA client."""
    st.set_page_config(page_title="Game Server Demo QA Client", layout="wide")
    init_state()
    render_sidebar()

    st.title("Game Server Demo QA Client")
    st.caption("Manual QA console for authenticated game sessions.")
    message_container = st.empty()
    render_game_catalog()
    render_current_session()
    render_known_sessions()
    render_message(message_container)


if __name__ == "__main__":
    main()
