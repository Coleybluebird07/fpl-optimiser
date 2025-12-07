import pandas as pd
import numpy as np
from fpl_api import get_bootstrap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Columns used as inputs to the ML model
FEATURE_COLUMNS = [
    "last_points",
    "form_last3",
    "form_last5",
    "season_avg",
    "home_avg",
    "fixture_difficulty",
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
    "premium_tag"
]


def load_dataset() -> pd.DataFrame:
    #Load the raw player-history CSV.
    return pd.read_csv("data/raw/player_history_sample.csv")


# ---------- Feature engineering ----------

def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    
    #Adds form-based and season-based features per player.
    df = df.sort_values(by=["player_id", "gameweek"])
    group = df.groupby("player_id")["total_points"]

    # Points from previous gameweek
    df["last_points"] = group.shift(1)

    # Rolling form windows
    df["form_last3"] = (
        group.rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["form_last5"] = (
        group.rolling(window=5, min_periods=1)
        .mean()
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

def add_cost_to_dataset(df):
    #Attach now_cost to each row from bootstrap player metadata.
    bootstrap = get_bootstrap()
    players = pd.DataFrame(bootstrap["elements"])[["id", "now_cost"]]
    players.rename(columns={"id": "player_id"}, inplace=True)
    
    df = df.merge(players, on="player_id", how="left")
    return df

def add_fixture_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    
    #Adds a 'fixture_difficulty' column based on how many points teams usually concede to FPL players. Higher = easier opponent.
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
    df["expected_points_baseline"] = df["form_last3"]
    return df

def evaluate_baseline(df: pd.DataFrame) -> float:
    df["abs_error"] = (df["total_points"] - df["expected_points_baseline"]).abs()
    return df["abs_error"].mean()


# ---------- ML model ----------

def train_ml_model(df: pd.DataFrame):
    
    # Train a linear regression model on the engineered features.
    features = FEATURE_COLUMNS

    df = df.dropna(subset=features + ["total_points"])

    X = df[features]
    y = df["total_points"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("\nModel coefficients (feature importance):")
    for feature_name, coef in zip(features, model.coef_):
        print(f"{feature_name:>16}: {coef:.3f}")

    return model, mae


def main():
    df = load_dataset()
    df = add_cost_to_dataset(df)  # ⭐ NEW
    df = add_form_features(df)
    df = add_fixture_difficulty(df)
    df = add_position_features(df)
    df = baseline_expected_points(df)

    baseline_mae = evaluate_baseline(df)
    print(f"Baseline MAE: {baseline_mae:.2f}")

    model, ml_mae = train_ml_model(df)
    print(f"Machine Learning MAE: {ml_mae:.2f}")


if __name__ == "__main__":
    main()