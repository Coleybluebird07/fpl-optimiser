from typing import List, Dict
import time
import pandas as pd

from src.fpl_api import get_bootstrap, get_player_history


def build_full_history_dataset() -> pd.DataFrame:

    #Fetches full history of gameweek data for all players.

    bootstrap = get_bootstrap()
    players = bootstrap["elements"]

    rows: List[Dict] = []

    for player in players:
        player_id = player["id"]
        player_name = f"{player['first_name']} {player['second_name']}"
        position = player["element_type"]  # 1=GK,2=DEF,3=MID,4=FWD
        team_id = player["team"]

        print(f"Fetching history for {player_name} ({player_id})...")

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

        time.sleep(0.2)

    df = pd.DataFrame(rows)
    return df


def main():
    df = build_full_history_dataset()
    print("Built dataset with", len(df), "rows")

    output_path = "data/raw/player_history_sample.csv"
    df.to_csv(output_path, index=False)
    print("Saved CSV to", output_path)


if __name__ == "__main__":
    main()