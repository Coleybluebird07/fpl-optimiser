import requests


BASE_URL = "https://fantasy.premierleague.com/api"


def get_bootstrap():
    # Fetches data as JSON.
    url = f"{BASE_URL}/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_player_history(player_id: int) -> dict:
    # Fetches detailed history for a single player.
    url = f"{BASE_URL}/element-summary/{player_id}/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # Simple manual test: run this file directly to see it working
    data = get_bootstrap()
    print("Top-level keys in bootstrap data:")
    for key in data.keys():
        print("-", key)