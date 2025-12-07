import requests

BASE_URL = "https://fantasy.premierleague.com/api"


def get_bootstrap():
    #Fetch main FPL game data (players, teams, events etc.) 
    url = f"{BASE_URL}/bootstrap-static/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_player_history(player_id: int) -> dict:
    #Fetch past performance history for players.
    url = f"{BASE_URL}/element-summary/{player_id}/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_fixtures():
    #Fetch all fixtures for the current season.
    url = f"{BASE_URL}/fixtures/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    #sanity check
    data = get_bootstrap()
    print("Teams:", len(data["teams"]))
    print("Players:", len(data["elements"]))