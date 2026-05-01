import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


# COMMON EVALUATION FUNCTION

def evaluate_model(model, X, y, target_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n===== {target_name.upper()} =====")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R2    : {r2:.4f}")

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)

    print(f"CV R2 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, X_test, y_test, y_pred


# PLOTTING FUNCTIONS

def plot_parity(y_test, y_pred, target_name):
    plt.figure(figsize=(4,3))
    plt.scatter(y_test, y_pred, edgecolors="black", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "--", color="black")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(target_name)
    plt.show()


def plot_residuals(y_pred, y_test):
    residuals = y_test - y_pred
    plt.figure(figsize=(4,3))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.show()


def plot_feature_importance(model, X, y):
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance = pd.Series(result.importances_mean, index=X.columns)
    importance.sort_values().tail(10).plot(kind="barh")
    plt.title("Top Features")
    plt.show()


# MODELS

def run_random_forest(X, y, name):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model, X_test, y_test, y_pred = evaluate_model(model, X, y, name)

    plot_parity(y_test, y_pred, name)
    plot_residuals(y_pred, y_test)
    plot_feature_importance(model, X_test, y_test)

    return model


def run_gradient_boosting(X, y, name):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )

    model, X_test, y_test, y_pred = evaluate_model(model, X, y, name)

    plot_parity(y_test, y_pred, name)
    plot_residuals(y_pred, y_test)
    plot_feature_importance(model, X_test, y_test)

    return model


def run_xgboost(X, y, name):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model, X_test, y_test, y_pred = evaluate_model(model, X, y, name)

    plot_parity(y_test, y_pred, name)
    plot_residuals(y_pred, y_test)
    plot_feature_importance(model, X_test, y_test)

    return model