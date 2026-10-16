"""CardioRisk AI - entraînement et évaluation des modèles prédictifs.

Ce script reproduit la partie « Machine Learning » du notebook, avec un pipeline
sans fuite d'information :

* l'imputation (Random Forest) et la standardisation sont **apprises uniquement
  sur les données d'entraînement** (elles font partie du Pipeline scikit-learn,
  donc refaites dans chaque pli de validation croisée) ;
* la profondeur de l'arbre est choisie par **validation croisée** (GridSearchCV)
  sur le jeu d'entraînement : le jeu de test n'est utilisé qu'une seule fois,
  pour l'évaluation finale ;
* les probabilités de l'arbre (entraîné avec ``class_weight='balanced'``) sont
  **calibrées** (CalibratedClassifierCV, méthode sigmoïde) avant d'être utilisées
  par l'application ;
* le modèle sauvegardé (standardisation + arbre calibré) **n'embarque pas
  l'imputeur Random Forest** : il pèse plusieurs centaines de Mo et l'application
  fournit toujours des valeurs complètes. Le fichier .pkl reste ainsi léger.

Usage :
    python train.py                       # lit data/heart_disease.csv
    python train.py --data chemin/vers/heart_disease.csv

Sorties : models/model_heart.pkl, models/feature_importances.json,
models/metrics.json et des figures dans docs/figures/.
"""

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "Heart_ stroke"  # nom exact de la colonne cible dans le CSV (avec l'espace)

FEATURES = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI",
            "heartRate", "glucose", "currentSmoker", "BPMeds",
            "prevalentHyp", "diabetes", "Gender_enc",
            "prevalentStroke_enc", "education_enc"]

EDUCATION_MAP = {"uneducated": 0, "primaryschool": 1,
                 "graduate": 2, "postgraduate": 3}

PARAM_GRID = {"clf__max_depth": [3, 4, 5, 6, 8, 10],
              "clf__min_samples_leaf": [5, 20]}


# ── Données ───────────────────────────────────────────────────────────────────
def _encode(series, mapping, name):
    """Encode une colonne texte avec un mapping explicite (insensible à la casse).

    Les valeurs manquantes restent NaN ; une modalité inconnue lève une erreur
    plutôt que d'être encodée silencieusement de travers.
    """
    cleaned = series.astype(str).str.strip().str.lower().where(series.notna())
    encoded = cleaned.map(mapping)
    unknown = sorted(cleaned[encoded.isna() & cleaned.notna()].unique())
    if unknown:
        raise ValueError(f"Modalités inattendues dans '{name}' : {unknown}")
    return encoded.astype(float)


def load_data(path):
    """Charge le CSV et renvoie (X, y). X garde ses valeurs manquantes."""
    df = pd.read_csv(path)
    df = df[df[TARGET].notna()].copy()

    df["Gender_enc"] = _encode(df["Gender"], {"male": 1, "female": 0}, "Gender")
    df["prevalentStroke_enc"] = _encode(df["prevalentStroke"],
                                        {"yes": 1, "no": 0}, "prevalentStroke")
    df["education_enc"] = _encode(df["education"], EDUCATION_MAP, "education")
    y = _encode(df[TARGET], {"yes": 1, "no": 0}, TARGET).astype(int)
    X = df[FEATURES].astype(float)
    return X, y


# ── Pipelines ─────────────────────────────────────────────────────────────────
def make_pipeline(classifier, seed=SEED, memory=None):
    """Imputation Random Forest -> standardisation -> classifieur.

    ``memory`` met en cache les étapes de prétraitement : l'imputation, coûteuse,
    n'est calculée qu'une fois par pli de validation croisée (et non une fois par
    combinaison d'hyperparamètres).
    """
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=seed, n_jobs=1),
        max_iter=5, random_state=seed)
    return Pipeline([("imputer", imputer),
                     ("scaler", StandardScaler()),
                     ("clf", classifier)], memory=memory)


def make_tree(seed=SEED, memory=None):
    return make_pipeline(DecisionTreeClassifier(class_weight="balanced",
                                                random_state=seed), seed, memory)


def make_mlp(seed=SEED, memory=None):
    return make_pipeline(MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.001, learning_rate_init=0.001, max_iter=500, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20),
        seed, memory)


# ── Évaluation ────────────────────────────────────────────────────────────────
SCORING = {"accuracy": "accuracy", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


def evaluate(model, X_test, y_test, cv_scores):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "cv_accuracy": cv_scores["test_accuracy"].mean(),
        "cv_accuracy_std": cv_scores["test_accuracy"].std(),
        "cv_recall": cv_scores["test_recall"].mean(),
        "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
    }


def run_experiment(data_path, seed=SEED, verbose=True):
    """Entraîne et évalue les deux modèles ; renvoie un dictionnaire de résultats."""
    with tempfile.TemporaryDirectory() as cache_dir:
        return _run(data_path, seed, verbose, joblib.Memory(cache_dir, verbose=0))


def _run(data_path, seed, verbose, memory):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if verbose:
        print(f"Split train/test : {len(X_train)} train | {len(X_test)} test "
              f"(cible positive : {y.mean() * 100:.1f} %)")

    # 1. Arbre de décision : profondeur choisie par validation croisée
    grid = GridSearchCV(make_tree(seed, memory), PARAM_GRID, cv=cv, scoring="roc_auc",
                        n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    tree = grid.best_estimator_
    if verbose:
        print(f"Meilleurs hyperparamètres (CV, AUC) : {grid.best_params_}")

    tree_cv = cross_validate(clone(tree), X_train, y_train, cv=cv,
                             scoring=SCORING, n_jobs=-1)
    tree_metrics = evaluate(tree, X_test, y_test, tree_cv)

    # 2. Modèle déployé : standardisation + arbre calibré, SANS imputeur.
    #    On impute X_train / X_test avec l'imputeur déjà appris sur l'entraînement.
    imputer = tree.named_steps["imputer"]
    X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=FEATURES,
                               index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES,
                              index=X_test.index)
    best_tree_params = {k.split("__", 1)[1]: v for k, v in grid.best_params_.items()}
    light_tree = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=seed,
                                       **best_tree_params))])
    tree_cal = CalibratedClassifierCV(light_tree, method="sigmoid", cv=cv)
    tree_cal.fit(X_train_imp, y_train)
    cal_proba = tree_cal.predict_proba(X_test_imp)[:, 1]
    calibration = {
        "brier_avant": tree_metrics["brier"],
        "brier_apres": brier_score_loss(y_test, cal_proba),
        "roc_auc_apres": roc_auc_score(y_test, cal_proba),
    }

    # 3. Réseau de neurones
    mlp = make_mlp(seed, memory)
    mlp_cv = cross_validate(clone(mlp), X_train, y_train, cv=cv,
                            scoring=SCORING, n_jobs=-1)
    mlp.fit(X_train, y_train)
    mlp_metrics = evaluate(mlp, X_test, y_test, mlp_cv)

    importances = pd.Series(tree.named_steps["clf"].feature_importances_,
                            index=FEATURES)
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_train_imp": X_train_imp, "X_test_imp": X_test_imp,
        "grid": grid, "tree": tree, "tree_cal": tree_cal, "mlp": mlp,
        "tree_pred": tree.predict(X_test), "mlp_pred": mlp.predict(X_test),
        "importances": importances,
        "metrics": {"Arbre de décision": tree_metrics,
                    "Réseau de neurones": mlp_metrics},
        "calibration": calibration,
    }


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_depth_curve(grid, save_to=None):
    res = pd.DataFrame(grid.cv_results_)
    leaf = grid.best_params_["clf__min_samples_leaf"]
    res = res[res["param_clf__min_samples_leaf"] == leaf].sort_values("param_clf__max_depth")
    depth = res["param_clf__max_depth"].astype(int)
    best = grid.best_params_["clf__max_depth"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depth, res["mean_train_score"], "b-o", label="AUC entraînement (CV)")
    ax.plot(depth, res["mean_test_score"], "r-s", label="AUC validation (CV)")
    ax.axvline(best, color="green", linestyle="--", label=f"Meilleure profondeur = {best}")
    ax.set_xlabel("Profondeur maximale")
    ax.set_ylabel("AUC ROC")
    ax.set_title(f"Arbre de décision : choix de la profondeur (min_samples_leaf={leaf})",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_confusions(res, save_to=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, key, cmap) in zip(axes, [("Arbre de décision", "tree_pred", "Blues"),
                                            ("Réseau de neurones", "mlp_pred", "Oranges")]):
        ConfusionMatrixDisplay(confusion_matrix(res["y_test"], res[key]),
                               display_labels=["No", "Yes"]).plot(
            ax=ax, colorbar=False, cmap=cmap)
        acc = res["metrics"][name]["accuracy"] * 100
        ax.set_title(f"{name}\n(accuracy test = {acc:.1f} %)", fontweight="bold")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_importances(importances, save_to=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Importance des variables (arbre de décision)", fontweight="bold")
    ax.set_xlabel("Importance (Gini)")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(metrics, save_to=None):
    keys = [("accuracy", "Accuracy\n(test)"), ("cv_accuracy", "Accuracy\n(CV 5 plis)"),
            ("recall", "Rappel\n(Yes)"), ("f1", "F1-score\n(Yes)"),
            ("roc_auc", "AUC ROC\n(test)")]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, (name, color) in zip([-width / 2, width / 2],
                                     [("Arbre de décision", "steelblue"),
                                      ("Réseau de neurones", "salmon")]):
        vals = [metrics[name][k] * 100 for k, _ in keys]
        bars = ax.bar(x + offset, vals, width, label=name, color=color, edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in keys])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title("Comparaison des modèles prédictifs", fontweight="bold", fontsize=13)
    ax.legend()
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_artifacts(res, root, figures=True):
    """Écrit le modèle déployé (léger), les importances, les métriques et les figures."""
    root = Path(root)
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(res["tree_cal"], models_dir / "model_heart.pkl")
    with open(models_dir / "feature_importances.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in res["importances"].items()}, f,
                  indent=2, ensure_ascii=False)

    def clean(d):
        return {k: round(float(v), 4) for k, v in d.items()}

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"modeles": {n: clean(m) for n, m in res["metrics"].items()},
                   "calibration": clean(res["calibration"]),
                   "meilleurs_hyperparametres": res["grid"].best_params_,
                   "scikit_learn": sklearn.__version__}, f, indent=2, ensure_ascii=False)

    if figures:
        fig_dir = root / "docs" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_depth_curve(res["grid"], fig_dir / "fig11_dt_depth_selection.png")
        plot_confusions(res, fig_dir / "fig12_confusion_matrices.png")
        plot_importances(res["importances"], fig_dir / "fig13_feature_importances.png")
        plot_comparison(res["metrics"], fig_dir / "fig15_models_comparison.png")
        plt.close("all")


def print_summary(res):
    print(f"\n{'Modèle':<22}{'Accuracy':>10}{'CV Acc':>10}{'Rappel':>10}"
          f"{'F1':>10}{'AUC':>10}")
    print("─" * 72)
    for name, m in res["metrics"].items():
        print(f"{name:<22}{m['accuracy'] * 100:>9.2f}%{m['cv_accuracy'] * 100:>9.2f}%"
              f"{m['recall'] * 100:>9.2f}%{m['f1'] * 100:>9.2f}%{m['roc_auc']:>10.3f}")
    c = res["calibration"]
    print(f"\nScore de Brier de l'arbre : {c['brier_avant']:.4f} avant calibration, "
          f"{c['brier_apres']:.4f} après (plus bas = mieux).")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(root / "data" / "heart_disease.csv"),
                        help="chemin du fichier heart_disease.csv")
    parser.add_argument("--no-figures", action="store_true",
                        help="ne pas générer les figures")
    args = parser.parse_args()

    plt.switch_backend("Agg")
    res = run_experiment(args.data)
    print_summary(res)
    save_artifacts(res, root, figures=not args.no_figures)
    size_mb = (root / "models" / "model_heart.pkl").stat().st_size / 1e6
    print(f"\nmodel_heart.pkl : {size_mb:.2f} Mo")
    print("Fichiers écrits dans models/ et docs/figures/.")


if __name__ == "__main__":
    main()
