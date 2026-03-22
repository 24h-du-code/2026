import json
import requests

API_BASE_URL = "https://24hcode2026.plaiades.fr/api"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 5796ca2ad9a9a8898df7d1540d3bec30"
}

def listGames():
    endpoint = "/list_games/"
    response = requests.get(url=API_BASE_URL+endpoint, headers=headers)
    return response.json()

def newGame(game_id):
    endpoint = "/newgame/"
    data = {"idgame": game_id}
    response = requests.post(url=API_BASE_URL+endpoint, headers=headers, data=json.dumps(data))
    return response.json()

def getState(session_id):
    endpoint = f"/get_state/?gamesessionid={session_id}"
    response = requests.get(url=API_BASE_URL+endpoint, headers=headers)
    return response.json()

def act(session_id, action):
    endpoint = "/act/"
    data = {"gamesessionid": session_id, "action": action}
    response = requests.post(url=API_BASE_URL+endpoint, headers=headers, data=json.dumps(data))
    return response.json()

def stopGame(session_id):
    endpoint = "/stop_game/"
    data = {"gamesessionid": session_id}
    response = requests.post(url=API_BASE_URL+endpoint, headers=headers, data=json.dumps(data))
    return response.json()