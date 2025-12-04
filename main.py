from src.fpl_api import get_bootstrap


def main():
    data = get_bootstrap()

    print("=== FPL Bootstrap Overview ===")
    print("Number of teams:", len(data["teams"]))
    print("Number of players:", len(data["elements"]))
    print("Number of gameweeks:", len(data["events"]))

    # Print first 5 player names as a sanity check
    print("\nFirst 5 players:")
    for player in data["elements"][:5]:
        name = f"{player['first_name']} {player['second_name']}"
        print("-", name)


if __name__ == "__main__":
    main()