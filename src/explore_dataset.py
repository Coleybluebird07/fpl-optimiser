import pandas as pd
import numpy as np

from src.fpl_api import get_bootstrap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Columns used as inputs to the ML model (global model)
FEATURE_COLUMNS = [
    "last_points",
    "form_last3",
    "form_last5",
    "minutes_last5",
    "starts_last5",
    "goals_last3",
    "goals_last5",
    "assists_last3",
    "assists_last5",
    "season_avg",
    "home_avg",
    "opp_att_strength",
    "opp_def_strength",
    "was_home",
    "minutes",
    "goals_scored",
    "assists",
    "is_gk",
    "is_def",
    "is_mid",
    "is_fwd",
    "points_per_90",
    "season_points_total",
    "season_points_per_appearance",
    "season_goal_rate",
    "season_assist_rate",
    "premium_tag",
]

# For now, each position uses the same feature set; we can specialise later.
FEATURE_COLUMNS_BY_POS = {
    1: FEATURE_COLUMNS,  # GK
    2: FEATURE_COLUMNS,  # DEF
    3: FEATURE_COLUMNS,  # MID
    4: FEATURE_COLUMNS,  # FWD
}


def load_dataset() -> pd.DataFrame:
    """Load the raw player-history CSV."""
    return pd.read_csv("data/raw/player_history_sample.csv")


# ---------- Feature engineering ----------

def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds form-based, minutes-based and season-based features per player.
    Uses cumulative season stats to avoid looking into the future.
    """
    df = df.sort_values(by=["player_id", "gameweek"]).copy()

    # Groups for different stats
    pts_group = df.groupby("player_id")["total_points"]
    minutes_group = df.groupby("player_id")["minutes"]
    goals_group = df.groupby("player_id")["goals_scored"]
    assists_group = df.groupby("player_id")["assists"]

    # Points from previous gameweek
    df["last_points"] = pts_group.shift(1)

    # Rolling form windows (points)
    df["form_last3"] = (
        pts_group.rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["form_last5"] = (
        pts_group.rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # "Start" = played 60+ minutes
    df["start_flag"] = (df["minutes"] >= 60).astype(int)
    starts_group = df.groupby("player_id")["start_flag"]

    df["minutes_last5"] = (
        minutes_group.rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["starts_last5"] = (
        starts_group.rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["goals_last3"] = (
        goals_group.rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["goals_last5"] = (
        goals_group.rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["assists_last3"] = (
        assists_group.rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["assists_last5"] = (
        assists_group.rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Average points this season (up to this row)
    df["season_avg"] = (
        df.groupby("player_id")["total_points"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Average points at home
    df["home_avg"] = (
        df[df["was_home"] == 1]
        .groupby("player_id")["total_points"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["home_avg"] = df["home_avg"].fillna(0)

    # Points per 90 mins in that match (rate)
    df["points_per_90"] = (
        (df["total_points"] / df["minutes"]) * 90
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # --- Season-long features (cumulative, no future leakage) ---

    df["appearances"] = (df["minutes"] > 0).astype(int)

    df = df.sort_values(["player_id", "gameweek"])
    group = df.groupby("player_id")

    # cumulative totals
    df["season_points_total"] = group["total_points"].cumsum()
    df["appearances_cum"] = group["appearances"].cumsum().replace(0, 1)

    df["season_points_per_appearance"] = (
        df["season_points_total"] / df["appearances_cum"]
    )

    df["season_goals_cum"] = group["goals_scored"].cumsum()
    df["season_assists_cum"] = group["assists"].cumsum()

    df["season_goal_rate"] = (
        df["season_goals_cum"] / df["appearances_cum"]
    )
    df["season_assist_rate"] = (
        df["season_assists_cum"] / df["appearances_cum"]
    )

    return df


def add_opponent_strength(latest: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds opponent defensive & attacking strength using full historical averages.

    - history_df is used to compute team-level stats
    - latest is where we attach opp_def_strength / opp_att_strength
      based on latest['opponent_team'].
    """
    latest = latest.copy()

    # Opponent attacking strength: avg goals scored per match
    team_att_strength = (
        history_df.groupby("team_id")["goals_scored"]
        .mean()
        .to_dict()
    )

    # Opponent defensive leakiness: avg FPL points conceded
    team_def_leak = (
        history_df.groupby("opponent_team")["total_points"]
        .mean()
    )
    max_leak = team_def_leak.max()
    team_def_strength = (team_def_leak / max_leak).to_dict()  # 0 = strong, 1 = weak

    latest["opp_att_strength"] = latest["opponent_team"].map(team_att_strength)
    latest["opp_def_strength"] = latest["opponent_team"].map(team_def_strength)

    # Fill missing with means so model always has something
    latest["opp_att_strength"].fillna(latest["opp_att_strength"].mean(), inplace=True)
    latest["opp_def_strength"].fillna(latest["opp_def_strength"].mean(), inplace=True)

    return latest


def add_cost_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Attach now_cost to each row from bootstrap player metadata."""
    bootstrap = get_bootstrap()
    players = pd.DataFrame(bootstrap["elements"])[["id", "now_cost"]]
    players.rename(columns={"id": "player_id"}, inplace=True)

    df = df.merge(players, on="player_id", how="left")
    df["now_cost"] = df["now_cost"] / 10.0
    return df


def add_fixture_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'fixture_difficulty' column from historical points conceded."""
    difficulty_map = (
        df.groupby("opponent_team")["total_points"]
        .mean()
        .to_dict()
    )
    df["fixture_difficulty"] = df["opponent_team"].map(difficulty_map)
    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds encoded position columns from 'position'."""
    df["is_gk"] = (df["position"] == 1).astype(int)
    df["is_def"] = (df["position"] == 2).astype(int)
    df["is_mid"] = (df["position"] == 3).astype(int)
    df["is_fwd"] = (df["position"] == 4).astype(int)
    df["premium_tag"] = (df["now_cost"] >= 11.0).astype(int)
    return df


# ---------- Targets & baseline model ----------

def add_future_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'target_next_points': next gameweek's points for each
    (player_id, gameweek) row. This is what we train the ML model on.
    """
    df = df.sort_values(["player_id", "gameweek"]).copy()
    df["target_next_points"] = (
        df.groupby("player_id")["total_points"].shift(-1)
    )
    return df


def baseline_expected_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple baseline: extrapolate from short-term form and minutes.
    """
    df = df.copy()
    df["expected_points_baseline"] = (
        df["form_last3"] * (df["minutes_last5"] / 90).fillna(1)
    )
    return df


def evaluate_baseline(df: pd.DataFrame) -> float:
    """
    Baseline error measured against *next* gameweek points.
    """
    df = df.copy()
    df = df[df["target_next_points"].notna()]
    df["abs_error"] = (df["target_next_points"] - df["expected_points_baseline"]).abs()
    return df["abs_error"].mean()


# ---------- ML models ----------

def _make_gbm():
    """Factory for a gradient boosting model tuned for MAE."""
    return GradientBoostingRegressor(
        loss="absolute_error",   # optimise MAE directly
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )


def train_ml_model(df: pd.DataFrame):
    """
    Train the global Gradient Boosting model on historical per-match FPL points.
    Also compute a calibration factor so that the average prediction matches
    the average true score on a held-out test set.
    """

    features = FEATURE_COLUMNS

    # Only keep rows where we have all features and a valid target
    df = df.dropna(subset=features + ["total_points"])

    X = df[features]
    y = df["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # ---- Calibration: rescale predictions so the mean matches reality ----
    if preds.mean() != 0:
        calibration = float(y_test.mean() / preds.mean())
    else:
        calibration = 1.0

    print("\nFeature importances (Gradient Boosting):")
    importances = model.feature_importances_
    for feature_name, importance in sorted(
        zip(features, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature_name:>24}: {importance:.3f}")

    print(f"\nUncalibrated ML MAE (next GW): {mae:.2f}")
    print(f"Calibration factor: {calibration:.2f}")

    return model, mae, calibration


def train_ml_model(df: pd.DataFrame):
    """
    Train the global Gradient Boosting model on historical per-match FPL points.

    We return:
      - model
      - MAE
      - a calibration factor that:
          * matches the mean prediction to the mean true value
          * and also shrinks the top tail so the 95th percentile of predictions
            is around a realistic FPL ceiling (~12 points).
    """

    features = FEATURE_COLUMNS

    # Only keep rows where we have all features and a valid target
    df = df.dropna(subset=features + ["total_points"])

    X = df[features]
    y = df["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # ---- Step 1: mean calibration ----
    if preds.mean() != 0:
        mean_calibration = float(y_test.mean() / preds.mean())
    else:
        mean_calibration = 1.0

    calibrated_preds = preds * mean_calibration

    # ---- Step 2: tail calibration (95th percentile) ----
    # Aim for 95th percentile of predictions ≈ 12 FPL points
    target_p95 = 12.0
    p95 = float(np.percentile(calibrated_preds, 95))

    if p95 > 0 and p95 > target_p95:
        tail_scale = target_p95 / p95
    else:
        tail_scale = 1.0

    final_calibration = mean_calibration * tail_scale

    print("\nFeature importances (Gradient Boosting):")
    importances = model.feature_importances_
    for feature_name, importance in sorted(
        zip(features, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature_name:>24}: {importance:.3f}")

    print(f"\nUncalibrated ML MAE (next GW): {mae:.2f}")
    print(f"Mean calibration factor: {mean_calibration:.2f}")
    print(f"Tail calibration factor: {tail_scale:.2f}")
    print(f"Final calibration factor: {final_calibration:.2f}")

    return model, mae, final_calibration


def walk_forward_validation(df: pd.DataFrame):
    """
    Time-series cross validation so the model cannot 'peek' at future data.
    Uses target_next_points as the label.
    """
    df = df.sort_values(["player_id", "gameweek"]).copy()
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_next_points"])

    mae_scores = []

    for gw in range(8, df["gameweek"].max() - 1):  # need next GW to exist
        train = df[df["gameweek"] <= gw]
        test = df[df["gameweek"] == gw + 1]

        if len(test) < 30:
            continue

        X_train = train[FEATURE_COLUMNS]
        y_train = train["target_next_points"]

        X_test = test[FEATURE_COLUMNS]
        y_test = test["target_next_points"]

        model = _make_gbm()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, preds))

    if mae_scores:
        print("\nWalk-forward MAE (next GW):", np.mean(mae_scores))
    else:
        print("\nWalk-forward validation skipped (not enough data).")


def main():
    # 1) Load and engineer features
    df = load_dataset()
    df = add_cost_to_dataset(df)
    df = add_form_features(df)
    df = add_opponent_strength(df, df)
    df = add_position_features(df)
    df = add_future_target(df)

    # 2) Baseline model
    df = baseline_expected_points(df)
    baseline_mae = evaluate_baseline(df)
    print(f"Baseline MAE (next GW): {baseline_mae:.2f}")

    # 3) Global ML model
    model, ml_mae = train_ml_model(df)

    # 4) Optional: walk-forward CV
    RUN_WALK_FORWARD = False
    if RUN_WALK_FORWARD:
        walk_forward_validation(df)

    return model, df


if __name__ == "__main__":
    main()