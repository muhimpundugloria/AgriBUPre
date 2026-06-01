import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = os.path.join("data", "agriculture_burundi.csv")
MODELS_DIR = os.path.join("models")
RANDOM_STATE = 42

NUM_FEATURES = [
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "utilisation_engrais",
    "acces_irrigation",
]

PROVINCES = [
    "Bubanza", "Bujumbura Rural", "Bururi", "Cankuzo", "Cibitoke", "Gitega",
    "Kayanza", "Kirundo", "Makamba", "Muramvya", "Muyinga", "Mwaro",
    "Ngozi", "Rutana", "Ruyigi"
]
REFERENCE_PROVINCE = "Bujumbura Rural"

CULTURES = ["Maïs", "Haricot", "Manioc", "Patate douce", "Sorgho", "Bananier"]
REFERENCE_CULTURE = "Bananier"

REFERENCE_SAISON = "A"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing target or leakage variables.
    df = df.dropna(subset=["bonne_recolte", "rendement_t_ha", "production_totale_t"]).reset_index(drop=True)

    # Impute pluvio by province/culture median, then province median, then global median.
    pluv_median = df["pluviometrie_mm"].median()

    def fill_pluv(row):
        if not np.isnan(row["pluviometrie_mm"]):
            return row["pluviometrie_mm"]
        mask = (
            (df["province"] == row["province"]) &
            (df["culture"] == row["culture"])
        )
        candidate = df.loc[mask, "pluviometrie_mm"].median()
        if not np.isnan(candidate):
            return candidate
        candidate = df.loc[df["province"] == row["province"], "pluviometrie_mm"].median()
        if not np.isnan(candidate):
            return candidate
        return pluv_median

    df["pluviometrie_mm"] = df.apply(fill_pluv, axis=1)

    # Impute utilisation_engrais by mode for culture/province, then culture, then global.
    overall_mode = int(df["utilisation_engrais"].dropna().mode().iloc[0])

    def fill_engrais(row):
        if not np.isnan(row["utilisation_engrais"]):
            return int(row["utilisation_engrais"])
        mask = (
            (df["province"] == row["province"]) &
            (df["culture"] == row["culture"])
        )
        candidate = df.loc[mask, "utilisation_engrais"].dropna()
        if not candidate.empty:
            return int(candidate.mode().iloc[0])
        candidate = df.loc[df["culture"] == row["culture"], "utilisation_engrais"].dropna()
        if not candidate.empty:
            return int(candidate.mode().iloc[0])
        return overall_mode

    df["utilisation_engrais"] = df.apply(fill_engrais, axis=1).astype(int)
    df["acces_irrigation"] = df["acces_irrigation"].astype(int)
    df["bonne_recolte"] = df["bonne_recolte"].astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUM_FEATURES].copy()

    # Saison one-hot encoding with A as reference.
    X["saison_B"] = (df["saison"] == "B").astype(int)

    # Province one-hot encoding with Bujumbura Rural as reference.
    for province in PROVINCES:
        if province == REFERENCE_PROVINCE:
            continue
        X[f"province_{province}"] = (df["province"] == province).astype(int)

    # Culture one-hot encoding with Bananier as reference.
    for culture in CULTURES:
        if culture == REFERENCE_CULTURE:
            continue
        X[f"culture_{culture}"] = (df["culture"] == culture).astype(int)

    feature_cols = NUM_FEATURES + ["saison_B"]
    feature_cols += [c for c in X.columns if c.startswith("province_") or c.startswith("culture_")]
    return X[feature_cols]


def train_models(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return dt, rf, lr


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    fpr, tpr, _ = roc_curve(y, y_prob)
    metrics["fpr"] = fpr.tolist()
    metrics["tpr"] = tpr.tolist()
    return metrics


def compute_overfitting_curve(X_train, X_test, y_train, y_test):
    depths = list(range(1, 21))
    train_scores = []
    test_scores = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        train_scores.append(float(model.score(X_train, y_train)))
        test_scores.append(float(model.score(X_test, y_test)))

    return {
        "depths": depths,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data()
    df = prepare_data(df)

    X = build_feature_matrix(df)
    y = df["bonne_recolte"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUM_FEATURES] = scaler.fit_transform(X_train[NUM_FEATURES])
    X_test_scaled[NUM_FEATURES] = scaler.transform(X_test[NUM_FEATURES])

    dt, rf, lr = train_models(X_train_scaled, y_train)

    dt_metrics = evaluate_model(dt, X_test_scaled, y_test)
    rf_metrics = evaluate_model(rf, X_test_scaled, y_test)
    lr_metrics = evaluate_model(lr, X_test_scaled, y_test)

    cv_scores = cross_val_score(
        rf, X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy"
    )

    overfitting = compute_overfitting_curve(X_train_scaled, X_test_scaled, y_train, y_test)

    feature_importances = {
        col: float(imp)
        for col, imp in zip(X.columns, rf.feature_importances_)
    }

    metrics = {
        "decision_tree": dt_metrics,
        "random_forest": {
            **rf_metrics,
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
        },
        "logistic_regression": lr_metrics,
    }

    metadata = {
        "feature_cols": X.columns.tolist(),
        "num_cols": NUM_FEATURES,
        "metrics": metrics,
        "rf_feature_importances": feature_importances,
        "overfitting": overfitting,
        "dataset_info": {
            "total_rows": int(len(df)),
            "provinces": int(len(PROVINCES)),
            "cultures": int(len(CULTURES)),
            "years": {
                "min": int(df["annee"].min()),
                "max": int(df["annee"].max()),
            },
        },
    }

    joblib.dump(dt, os.path.join(MODELS_DIR, "decision_tree.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("Models and metadata saved in models/")
    print("Dataset used for training:", len(df), "rows")
    print("Random Forest CV accuracy: %.3f ± %.3f" % (cv_scores.mean(), cv_scores.std()))


if __name__ == "__main__":
    main()
