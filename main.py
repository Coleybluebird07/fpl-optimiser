from src.fpl_api import get_bootstrap, get_player_history
from src.build_dataset import build_sample_history_dataset


def main():
    df = build_sample_history_dataset(num_players=10)
    print("Built dataset with", len(df), "rows")

    # Save to data/raw folder
    output_path = "data/raw/player_history_sample.csv"
    # Make sure folder exists (we'll create it manually for now)
    df.to_csv(output_path, index=False)
    print("Saved CSV to", output_path)


if __name__ == "__main__":
    main()