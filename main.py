from src.fpl_api import get_bootstrap, get_player_history


def main():
    data = get_bootstrap()

    print("=== FPL Bootstrap Overview ===")
    print("Number of teams:", len(data["teams"]))
    print("Number of players:", len(data["elements"]))
    print("Number of gameweeks:", len(data["events"]))

    first_player = data["elements"][0]
    player_id = first_player["id"]
    player_name = f"{first_player['first_name']} {first_player['second_name']}"

    print(f"\nFetching history for {player_name} (ID: {player_id})")
    history_data = get_player_history(player_id)

    history = history_data["history"]
    print("Past gameweeks:", len(history))

    print("\nFirst few matches:")
    for gw in history[:3]:
        print(
            f"GW {gw['round']}: {gw['total_points']} pts, "
            f"{gw['minutes']} mins vs team {gw['opponent_team']}"
        )


if __name__ == "__main__":
    main()