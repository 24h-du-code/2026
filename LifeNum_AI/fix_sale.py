import json
import requests
from time import sleep

def fixSale():
    URL = "https://24hcode2026.plaiades.fr/api"
    GAMES_LIST = [4]

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 5796ca2ad9a9a8898df7d1540d3bec30"
    }

    for i in GAMES_LIST:
        payload = {"idgame": i}
        res = requests.post(url=URL+"/newgame/", headers=headers, json=payload)
        data = res.json()
        id_game = data['existing_session_id'] if res.status_code == 409 else data['gamesessionid']
        payload = {"gamesessionid": id_game}
        stop_res = requests.post(url=URL+"/stop_game/", headers=headers, json=payload)
        sleep(1)

fixSale()