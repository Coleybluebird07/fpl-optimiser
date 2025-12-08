import os
import numpy as np
import pandas as pd

from src.fpl_api import get_bootstrap, get_fixtures
from src.explore_dataset import (
    load_dataset,
    add_form_features,
    add_opponent_strength,
    add_position_features,
    train_ml_model,
    FEATURE_COLUMNS,
    add_cost_to_dataset,
    baseline_expected_points,
)

# Minimum number of recent starts we tolerate
# If starts_last5 <= MIN_STARTS_THRESHOLD we treat them as strong rotation risk
MIN_STARTS_THRESHOLD = 1

# Hard sanity cap on single-GW expected points
MAX_PREDICTED_POINTS = 13.0


# ---------------------------------------------------------------------
#  Training / feature-building pipeline
# ---------------------------------------------------------------------
def build_model_and_features():
    """
    Build full historical dataframe with engineered features and train the
    global ML model. Returns the dataframe, the model, and a calibration factor.
    """
    df = load_dataset()
    df = add_cost_to_dataset(df)
    df = add_form_features(df)
    df = add_opponent_strength(df, df)
    df = add_position_features(df)

    model, ml_mae, calibration = train_ml_model(df)
    print(f"\nTrained global model. Calibration factor: {calibration:.2f}")
    return df, model, calibration


# ---------------------------------------------------------------------
#  Fixture utilities
# ---------------------------------------------------------------------
def get_latest_state_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, take their most recent gameweek row as their 'current state'.
    """
    df_sorted = df.sort_values(["player_id", "gameweek"])
    latest = df_sorted.groupby("player_id").tail(1).copy()
    return latest


def get_next_gameweek_fixture_map():
    """
    Build dict: team_id -> (opponent_team_id, was_home_flag)
    for the next upcoming gameweek.
    """
    fixtures = get_fixtures()
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
    """
    Overwrites opponent_team, was_home and fixture_difficulty in
    latest player rows so they reflect the *next fixture*.
    """
    latest = latest.copy()

    fixture_map, next_gw = get_next_gameweek_fixture_map()
    print(f"\nApplying fixture context for GW {next_gw}...")

    # Build difficulty map from history (avg FPL points conceded)
    difficulty_map = (
        full_df.groupby("opponent_team")["total_points"]
        .mean()
        .to_dict()
    )

    def map_opp(team_id):
        return fixture_map.get(team_id, (None, None))[0]

    def map_home(team_id):
        return fixture_map.get(team_id, (None, None))[1]

    latest["opponent_team"] = latest["team_id"].map(map_opp)
    latest["was_home"] = latest["team_id"].map(map_home)

    # New fixture difficulty based on next opponent
    latest["fixture_difficulty"] = latest["opponent_team"].map(difficulty_map)

    return latest


def add_current_cost(latest: pd.DataFrame) -> pd.DataFrame:
    """
    Add current live FPL price (now_cost is provided in tenths of millions).
    """
    latest = latest.copy()
    bootstrap = get_bootstrap()
    elements = bootstrap["elements"]

    id_to_cost = {p["id"]: p["now_cost"] / 10.0 for p in elements}
    latest["now_cost"] = latest["player_id"].map(id_to_cost)
    return latest


# ---------------------------------------------------------------------
#  Rotation / start-likelihood adjustment
# ---------------------------------------------------------------------
def apply_rotation_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Down-weight expected points for players who haven't been starting regularly.

    This is intentionally strict: we prefer to miss the odd punt rather than
    recommend non-starters.
    """
    df = df.copy()

    # Approximate start likelihood from the last 5 matches
    df["start_likelihood"] = df["starts_last5"] / 5.0

    # Hard hammer for players with virtually no recent starts
    zero_start_mask = df["starts_last5"] <= MIN_STARTS_THRESHOLD
    df.loc[zero_start_mask, "predicted_points"] *= 0.15  # 85% penalty

    # Softer scaling across the board: 0.6x–1.2x depending on recent starts
    scale = 0.6 + 0.6 * df["start_likelihood"]  # from 0.6 (never) to 1.2 (always)
    df["predicted_points"] *= scale

    return df


# ---------------------------------------------------------------------
#  Main prediction pipeline
# ---------------------------------------------------------------------
def predict_next_gameweek():
    """
    1) Train + build enriched historical DF
    2) Take latest observation per player
    3) Apply next fixture context
    4) Predict with calibrated global model
    5) Match prediction distribution to historical (95th percentile)
    6) Blend ML prediction with baseline form model
    7) Apply rotation penalties
    8) Clip to realistic FPL range
    9) Save and print top players
    """
    full_df, model, calibration = build_model_and_features()

    # -----------------------
    # Historical distribution (for scaling)
    # -----------------------
    full_X = full_df[FEATURE_COLUMNS].fillna(0)
    full_preds = model.predict(full_X) * calibration

    # Compare 95th percentile of true vs predicted
    true_p95 = float(np.percentile(full_df["total_points"], 95))
    pred_p95 = float(np.percentile(full_preds, 95))

    if pred_p95 > 0:
        dist_scale = true_p95 / pred_p95
    else:
        dist_scale = 1.0

    print(f"\nHistorical 95th percentile true points: {true_p95:.2f}")
    print(f"Historical 95th percentile predicted:   {pred_p95:.2f}")
    print(f"Distribution scale factor:             {dist_scale:.2f}")

    # -----------------------
    # Current state for next GW
    # -----------------------
    latest = get_latest_state_per_player(full_df)
    latest = apply_next_fixture_context(latest, full_df)
    latest = add_current_cost(latest)

    # Add baseline expected_points_baseline (form_last3 * minutes_last5/90)
    latest = baseline_expected_points(latest)

    # Base ML predictions for next GW
    X_latest = latest[FEATURE_COLUMNS].fillna(0)
    ml_pred = model.predict(X_latest)

    # Apply calibration + distribution scaling
    ml_pred = ml_pred * (calibration * dist_scale)

    # -----------------------
    # Hybrid prediction: ML + baseline
    # -----------------------
    alpha = 0.65  # weight for ML model; (1 - alpha) for baseline
    baseline_pred = latest["expected_points_baseline"]

    latest["predicted_points"] = alpha * ml_pred + (1.0 - alpha) * baseline_pred

    # Rotation / start likelihood penalty
    latest = apply_rotation_penalty(latest)

    # Clip to sane range for a single GW
    latest["predicted_points"] = latest["predicted_points"].clip(
        lower=0.0, upper=MAX_PREDICTED_POINTS
    )

    # Prepare output
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

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_path = "data/predictions/next_gw_predictions.csv"
    output.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")

    # Preview
    print("\nTop 15 players by predicted points:")
    print(output.head(15))


def main():
    predict_next_gameweek()


if __name__ == "__main__":
    main()