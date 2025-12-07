import os
import pandas as pd

from src.fpl_api import get_bootstrap, get_fixtures
from explore_dataset import (
    load_dataset,
    add_form_features,
    add_fixture_difficulty,
    add_position_features,
    train_ml_model,
    FEATURE_COLUMNS,
    add_cost_to_dataset,
)


def build_model_and_features():

    # Rebuild the full feature dataframe and train the ML model.
    df = load_dataset()
    df = add_cost_to_dataset(df) 
    df = add_form_features(df)
    df = add_fixture_difficulty(df)
    df = add_position_features(df)

    model, ml_mae = train_ml_model(df)
    print(f"\nTrained model, ML MAE: {ml_mae:.2f}")
    return df, model


def get_latest_state_per_player(df: pd.DataFrame) -> pd.DataFrame:
  
    # For each player, take their most recent gameweek row as their 'current state'.
    df_sorted = df.sort_values(["player_id", "gameweek"])
    latest = df_sorted.groupby("player_id").tail(1).copy()
    return latest


def get_next_gameweek_fixture_map():
    
    #Build a dict: team_id -> (opponent_team_id, was_home) for the next upcoming gameweek.
    fixtures = get_fixtures()

    # Only fixtures that belong to a GW and are not finished
    upcoming = [f for f in fixtures if f["event"] is not None and not f["finished"]]
    if not upcoming:
        raise RuntimeError("No upcoming fixtures found")

    next_gw = min(f["event"] for f in upcoming)
    next_games = [f for f in upcoming if f["event"] == next_gw]

    fixture_map = {}
    for f in next_games:
        home = f["team_h"]
        away = f["team_a"]
        fixture_map[home] = (away, 1)   # 1 = home
        fixture_map[away] = (home, 0)   # 0 = away

    return fixture_map, next_gw


def apply_next_fixture_context(latest: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    
    #Overwrite 'opponent_team', 'was_home' and 'fixture_difficulty' in the latest rows to reflect the *next* fixture instead of the last one.
    fixture_map, next_gw = get_next_gameweek_fixture_map()
    print(f"\nApplying fixture context for GW {next_gw}...")

    # Build difficulty map from full history
    difficulty_map = (
        full_df.groupby("opponent_team")["total_points"]
        .mean()
        .to_dict()
    )

    def map_opp(team_id):
        opp, _ = fixture_map.get(team_id, (None, None))
        return opp

    def map_home(team_id):
        _, home_flag = fixture_map.get(team_id, (None, None))
        return home_flag

    latest["opponent_team"] = latest["team_id"].map(map_opp)
    latest["was_home"] = latest["team_id"].map(map_home)

    # New fixture difficulty based on next opponent
    latest["fixture_difficulty"] = latest["opponent_team"].map(difficulty_map)

    return latest


def add_current_cost(latest: pd.DataFrame) -> pd.DataFrame:

    # Add current FPL price.
    bootstrap = get_bootstrap()
    elements = bootstrap["elements"]
    id_to_cost = {p["id"]: p["now_cost"] / 10.0 for p in elements}
    latest["now_cost"] = latest["player_id"].map(id_to_cost)
    return latest


def predict_next_gameweek():
    # 1. Train model & get full history DF with features
    full_df, model = build_model_and_features()

    # 2. Get latest state for each player
    latest = get_latest_state_per_player(full_df)

    # 3. Apply next GW fixture context
    latest = apply_next_fixture_context(latest, full_df)

    # 4. Add current cost information
    latest = add_current_cost(latest)

    # 5. Prepare features and predict
    X_latest = latest[FEATURE_COLUMNS].fillna(0)
    latest["predicted_points"] = model.predict(X_latest)

    # 6. Select output columns
    output = latest[
        [
            "player_id",
            "player_name",
            "position",
            "team_id",
            "now_cost",
            "predicted_points",
        ]
    ].copy()

    output = output.sort_values("predicted_points", ascending=False)

    # 7. Save and preview
    os.makedirs("data/predictions", exist_ok=True)
    out_path = "data/predictions/next_gw_predictions.csv"
    output.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")

    print("\nTop 15 players by predicted points:")
    print(output.head(15))


def main():
    predict_next_gameweek()


if __name__ == "__main__":
    main()