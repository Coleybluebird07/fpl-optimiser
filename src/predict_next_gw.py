import os
import pandas as pd

from explore_dataset import (
    load_dataset,
    add_form_features,
    add_fixture_difficulty,
    add_position_features,
    train_ml_model,
    FEATURE_COLUMNS,
)

from fpl_api import get_bootstrap


def build_model_and_features():
    df = load_dataset()
    df = add_form_features(df)
    df = add_fixture_difficulty(df)
    df = add_position_features(df)

    model, ml_mae = train_ml_model(df)
    print(f"\nTrained model, ML MAE: {ml_mae:.2f}")

    return df, model


def get_latest_state_per_player(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(["player_id", "gameweek"])
    latest = df_sorted.groupby("player_id").tail(1).copy()
    return latest


def add_current_cost(latest: pd.DataFrame) -> pd.DataFrame:
    #Fetch current player prices 

    bootstrap = get_bootstrap()
    elements = bootstrap["elements"]

    id_to_cost = {p["id"]: p["now_cost"] / 10.0 for p in elements}

    latest["now_cost"] = latest["player_id"].map(id_to_cost)
    return latest


def predict_next_gameweek():
    df, model = build_model_and_features()

    latest = get_latest_state_per_player(df)

    # 3) Build feature matrix for prediction
    X_latest = latest[FEATURE_COLUMNS].fillna(0)

    # 4) Predict expected points
    latest["predicted_points"] = model.predict(X_latest)

    # 5) Add current FPL price
    latest = add_current_cost(latest)

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

    os.makedirs("data/predictions", exist_ok=True)
    output_path = "data/predictions/next_gw_predictions.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

    # Show top 15
    print("\nTop 15 players by predicted points:")
    print(output.head(15))


def main():
    predict_next_gameweek()


if __name__ == "__main__":
    main()