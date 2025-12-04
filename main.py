from src.fpl_api import get_bootstrap, get_player_history


def main():
    data = get_bootstrap()
    
    # Take the first player and fetch their history
    first_player = data["elements"][0]
    player_id = first_player["id"]
    player_name = f"{first_player['first_name']} {first_player['second_name']}"

    print(f"\nFetching history for player: {player_name} (ID: {player_id})")
    history_data = get_player_history(player_id)

    # 'history' is a list of gameweek performances
    history = history_data["history"]
    print(f"Number of past gameweeks for this player: {len(history)}")

    # Let's print the first 3 gameweeks for this player
    print("\nFirst 3 gameweeks for this player:")
    for gw in history[:3]:
        print(
            f"GW {gw['round']}: {gw['total_points']} pts, "
            f"{gw['minutes']} mins, opponent_team={gw['opponent_team']}"
        )


if __name__ == "__main__":
    main()