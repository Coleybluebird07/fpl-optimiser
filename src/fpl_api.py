import requests

BASE_URL = "https://fantasy.premierleague.com/api"

def get_bootstrap():
    """
    Fetches main FPL game data (players, teams, events etc.)
    """
    url = f"{BASE_URL}/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_player_history(player_id: int) -> dict:
    """
    Fetches past performance history for a specific player.
    """
    url = f"{BASE_URL}/element-summary/{player_id}/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    data = get_bootstrap()
    print("Top-level keys:")
    for key in data.keys():
        print("-", key)