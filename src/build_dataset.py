from typing import List, Dict
import pandas as pd
from src.fpl_api import get_bootstrap, get_player_history

def build_sample_history_dataset(num_players: int = 10) -> pd.DataFrame:
    #Build a sample dataset of player gameweek histories for the first `num_players`.
    bootstrap = get_bootstrap()
    players = bootstrap["elements"]

    rows: List[Dict] = []

    # We only take the first `num_players` to avoid hammering the API while testing
    for player in players[:num_players]:
        player_id = player["id"]
        player_name = f"{player['first_name']} {player['second_name']}"
        position = player["element_type"]   # 1=GK, 2=DEF, 3=MID, 4=FWD
        team_id = player["team"] #Change to team name at some point

        print(f"Fetching history for {player_name} (ID: {player_id})...")

        history_data = get_player_history(player_id)
        history = history_data["history"]

        for gw in history:
            row = {
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "team_id": team_id,
                "gameweek": gw["round"],
                "total_points": gw["total_points"],
                "minutes": gw["minutes"],
                "was_home": gw["was_home"],
                "opponent_team": gw["opponent_team"],
                "goals_scored": gw["goals_scored"],
                "assists": gw["assists"],
                "clean_sheets": gw["clean_sheets"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df