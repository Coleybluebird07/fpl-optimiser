import pandas as pd
import numpy as np
from src.fpl_api import get_bootstrap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Columns used as inputs to the ML model
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
    "opp_def_strength",    
    "opp_att_strength",    
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


def load_dataset() -> pd.DataFrame:
    # Load the raw player-history CSV.
    return pd.read_csv("data/raw/player_history_sample.csv")


# ---------- Feature engineering ----------

def add_form_features(df: pd.DataFrame) -> pd.DataFrame:

    #Adds form-based, minutes-based and season-based features per player.
    df = df.sort_values(by=["player_id", "gameweek"])

    # Groups for different stats
    pts_group = df.groupby("player_id")["total_points"]
    minutes_group = df.groupby("player_id")["minutes"]
    goals_group = df.groupby("player_id")["goals_scored"]
    assists_group = df.groupby("player_id")["assists"]

    # Points from previous gameweek
    df["last_points"] = pts_group.shift(1)

    # Rolling form windows
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

    # Average points this season
    df["season_avg"] = (
        df.groupby("player_id")["total_points"]
        .transform("mean")
    )

    # Average points at home
    df["home_avg"] = (
        df[df["was_home"] == 1]
        .groupby("player_id")["total_points"]
        .transform("mean")
    )
    df["home_avg"] = df["home_avg"].fillna(0)

    # Points per 90 mins in that match (rate)
    df["points_per_90"] = (
        (df["total_points"] / df["minutes"]) * 90
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # --- Season-long features ---

    df["season_points_total"] = (
        df.groupby("player_id")["total_points"]
        .transform("sum")
    )

    df["appearances"] = (
        (df["minutes"] > 0).astype(int)
    )

    df["season_points_per_appearance"] = (
        df.groupby("player_id")["total_points"]
        .transform("sum")
        /
        df.groupby("player_id")["appearances"].transform("sum").replace(0, 1)
    )

    df["season_goal_rate"] = (
        df.groupby("player_id")["goals_scored"]
        .transform("sum")
        /
        df.groupby("player_id")["appearances"].transform("sum").replace(0, 1)
    )

    df["season_assist_rate"] = (
        df.groupby("player_id")["assists"]
        .transform("sum")
        /
        df.groupby("player_id")["appearances"].transform("sum").replace(0, 1)
    )

    return df

def add_future_target(df: pd.DataFrame) -> pd.DataFrame:
    # Adds a 'target_next_points' column: next gameweek's points

    df = df.sort_values(["player_id", "gameweek"]).copy()
    df["target_next_points"] = (
        df.groupby("player_id")["total_points"].shift(-1)
    )
    return df


def add_opponent_strength(latest: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    #Adds opponent defensive & attacking strength using full historical averages.

    latest = latest.copy()

    # 1) Opponent attacking strength: average goals scored per match
    team_att_strength = (
        history_df.groupby("team_id")["goals_scored"]
        .mean()
        .to_dict()
    )

    # 2) Opponent defensive leakiness: average FPL points conceded
    team_def_leak = (
        history_df.groupby("opponent_team")["total_points"]
        .mean()
    )
    max_leak = team_def_leak.max()
    team_def_strength = (team_def_leak / max_leak).to_dict()  # higher = weaker defence

    # Map strengths onto latest using opponent_team
    latest["opp_att_strength"] = latest["opponent_team"].map(team_att_strength)
    latest["opp_def_strength"] = latest["opponent_team"].map(team_def_strength)

    # Fill missing values with column means (after mapping)
    latest["opp_att_strength"].fillna(latest["opp_att_strength"].mean(), inplace=True)
    latest["opp_def_strength"].fillna(latest["opp_def_strength"].mean(), inplace=True)

    return latest


def add_cost_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Attach now_cost to each row from bootstrap.
    bootstrap = get_bootstrap()
    players = pd.DataFrame(bootstrap["elements"])[["id", "now_cost"]]
    players.rename(columns={"id": "player_id"}, inplace=True)

    df = df.merge(players, on="player_id", how="left")
    return df


def add_fixture_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    difficulty_map = (
        df.groupby("opponent_team")["total_points"]
        .mean()
        .to_dict()
    )
    df["fixture_difficulty"] = df["opponent_team"].map(difficulty_map)
    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds encoded position columns from 'position'.
    df["is_gk"] = (df["position"] == 1).astype(int)
    df["is_def"] = (df["position"] == 2).astype(int)
    df["is_mid"] = (df["position"] == 3).astype(int)
    df["is_fwd"] = (df["position"] == 4).astype(int)
    df["premium_tag"] = (df["now_cost"] >= 11.0).astype(int)

    return df


# ---------- Baseline model ----------

def baseline_expected_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["expected_points_baseline"] = (
        df["form_last3"] * (df["minutes_last5"] / 90).fillna(1)
    )
    return df


def evaluate_baseline(df: pd.DataFrame) -> float:
    df = df.copy()
    # Only rows where we actually know next GW's points
    df = df[df["target_next_points"].notna()]
    df["abs_error"] = (
        df["target_next_points"] - df["expected_points_baseline"]
    ).abs()
    return df["abs_error"].mean()


# ---------- ML model ----------

def train_ml_model(df: pd.DataFrame):
    features = FEATURE_COLUMNS
    
    df = df.dropna(subset=features + ["target_next_points"])

    X = df[features]
    y = df["target_next_points"]

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

    print("\nFeature importances (Gradient Boosting):")
    importances = model.feature_importances_
    for feature_name, importance in sorted(
        zip(features, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{feature_name:>24}: {importance:.3f}")

    return model, mae

def walk_forward_validation(df: pd.DataFrame):
    df = df.sort_values(["player_id", "gameweek"])
    mae_scores = []

    features = FEATURE_COLUMNS
    df = df.dropna(subset=features + ["total_points"])

    for gw in range(8, df["gameweek"].max()):  # start validating after GW8
        train = df[df["gameweek"] <= gw]
        test = df[df["gameweek"] == gw + 1]

        if len(test) < 30:
            continue

        X_train = train[features]
        y_train = train["total_points"]

        X_test = test[features]
        y_test = test["total_points"]

        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, preds))

    print("\nWalk-forward MAE:", np.mean(mae_scores))


def main():
    # 1) Load features
    df = load_dataset()
    df = add_cost_to_dataset(df)
    df = add_form_features(df)
    df = add_opponent_strength(df, df)
    df = add_position_features(df)
    df = add_future_target(df)      

    # 2) Baseline model
    df = baseline_expected_points(df)
    baseline_mae = evaluate_baseline(df)
    print(f"Baseline MAE: {baseline_mae:.2f}")

    # 3) ML model
    model, ml_mae = train_ml_model(df)
    print(f"Machine Learning MAE: {ml_mae:.2f}")

    # 4) TS Validations
    walk_forward_validation(df)
    return model, df


if __name__ == "__main__":
    main()