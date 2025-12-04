import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def load_dataset():
    df = pd.read_csv("data/raw/player_history_sample.csv")
    return df


def add_form_feature(df):
    df = df.sort_values(by=["player_id", "gameweek"])
    df["form_last3"] = (
        df.groupby("player_id")["total_points"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_position_features(df):
    df["is_gk"] = (df["position"] == 1).astype(int)
    df["is_def"] = (df["position"] == 2).astype(int)
    df["is_mid"] = (df["position"] == 3).astype(int)
    df["is_fwd"] = (df["position"] == 4).astype(int)
    return df


def baseline_expected_points(df):
    df["expected_points_baseline"] = df["form_last3"]
    return df


def evaluate_baseline(df):
    df["abs_error"] = (df["total_points"] - df["expected_points_baseline"]).abs()
    return df["abs_error"].mean()


def train_ml_model(df):
    features = [
        "form_last3", "minutes", "goals_scored", "assists",
        "is_gk", "is_def", "is_mid", "is_fwd"
    ]

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
        print(f"  {feature_name:>12}: {coef:.3f}")

    return model, mae


def main():
    df = load_dataset()
    df = add_form_feature(df)
    df = add_position_features(df)
    df = baseline_expected_points(df)

    baseline_mae = evaluate_baseline(df)
    print(f"Baseline MAE: {baseline_mae:.2f}")

    model, ml_mae = train_ml_model(df)
    print(f"Machine Learning MAE: {ml_mae:.2f}")


if __name__ == "__main__":
    main()