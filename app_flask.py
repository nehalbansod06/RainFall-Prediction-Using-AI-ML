import base64
import io
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


app = Flask(__name__)
sns.set_theme(style="whitegrid", context="talk")


# =========================
# Browser / API Settings
# =========================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# =========================
# App State
# =========================
LATEST_ARTIFACT: Dict[str, object] = {}

# Labels that mean "rain" or "no rain" in different datasets.
POSITIVE_LABELS = {"yes", "1", "true", "rain", "rainy", "y", "positive"}
NEGATIVE_LABELS = {"no", "0", "false", "dry", "n", "negative"}
# Common names we check for when trying to find the target column automatically.
TARGET_NAME_HINTS = {"rainfall", "target", "label", "class", "outcome", "result", "response"}


# =========================
# Data Helpers
# =========================
def clean_columns(columns: List[str]) -> List[str]:
    return [str(column).strip().replace("\ufeff", "") for column in columns]


# Read uploaded data files from CSV or Excel.
def load_dataframe(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# Make the data easier to work with by cleaning headers and converting numbers.
def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = clean_columns(cleaned.columns.tolist())
    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = pd.to_numeric(cleaned[column], errors="ignore")
    return cleaned


# Try to guess which column contains the answer the model should learn.
def pick_target_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        lowered = str(column).strip().lower()
        if any(hint == lowered or hint in lowered for hint in TARGET_NAME_HINTS):
            return column
    if len(df.columns) >= 2:
        return df.columns[-1]
    return None


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    # Fill missing values, scale numbers, and turn text categories into model-friendly columns.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


# =========================
# Model Setup
# =========================
def build_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    }


# Small parameter grids used when the user turns tuning on.
def build_model_grids() -> Dict[str, Dict[str, List[object]]]:
    return {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 3.0, 10.0]},
        "Random Forest": {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [None, 8, 16],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
        },
        "SVM (RBF)": {
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__gamma": ["scale", "auto"],
        },
    }


# Wrap preprocessing and the classifier together so training and prediction use the same steps.
def make_pipeline(estimator, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", estimator),
        ]
    )


# Turn encoded model columns back into readable feature names for charts.
def encode_feature_names(preprocessor: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    feature_names = list(numeric_cols)
    if categorical_cols and "cat" in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_["cat"]
        onehot = cat_transformer.named_steps.get("onehot")
        if onehot is not None:
            feature_names.extend(onehot.get_feature_names_out(categorical_cols).tolist())
    return feature_names


# =========================
# Chart Styling
# =========================
def apply_figure_style(fig) -> None:
    fig.patch.set_facecolor("#0b1320")
    for axis in fig.axes:
        axis.set_facecolor("#101a2b")
        axis.title.set_color("#eef5ff")
        axis.xaxis.label.set_color("#d7e4f2")
        axis.yaxis.label.set_color("#d7e4f2")
        axis.tick_params(colors="#c7d6e6")
        for spine in axis.spines.values():
            spine.set_color("#35506d")


# Save a chart as a base64 image string so it can be sent in JSON.
def fig_to_uri(fig) -> str:
    apply_figure_style(fig)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("utf-8")


def safe_metric_score(metric_fn, y_true, y_pred_or_prob):
    try:
        return metric_fn(y_true, y_pred_or_prob)
    except ValueError:
        return None


# =========================
# Label Handling
# =========================
def parse_target_labels(target_series: pd.Series) -> Tuple[pd.Series, pd.Index]:
    cleaned = target_series.dropna()
    if cleaned.empty:
        raise ValueError("The target column does not contain any usable labels.")

    normalized = cleaned.astype(str).str.strip().str.lower()
    normalized = normalized.replace({"": np.nan, "nan": np.nan, "none": np.nan}).dropna()

    unique_values = normalized.unique().tolist()
    if len(unique_values) < 2:
        raise ValueError("The target column must contain at least two classes.")
    if len(unique_values) > 2:
        preview = ", ".join(map(str, unique_values[:6]))
        raise ValueError(f"The target column must be binary. Found classes: {preview}.")

    encoded, _ = pd.factorize(normalized)
    return pd.Series(encoded.astype(int), index=normalized.index), normalized.index


# =========================
# Form Helpers
# =========================
def build_feature_defaults(feature_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for column in feature_df.columns:
        if column in numeric_cols:
            series = pd.to_numeric(feature_df[column], errors="coerce")
            defaults[column] = float(series.median()) if series.notna().any() else 0.0
        else:
            series = feature_df[column].astype(str).replace("nan", np.nan).dropna()
            defaults[column] = str(series.mode(dropna=True).iloc[0]) if not series.empty else "Unknown"
    return defaults


# Pick a few useful category values so the UI can show dropdowns.
def build_category_options(feature_df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, List[str]]:
    options: Dict[str, List[str]] = {}
    for column in categorical_cols:
        series = feature_df[column].astype(str).replace("nan", np.nan).dropna()
        options[column] = series.value_counts().head(12).index.astype(str).tolist() or ["Unknown"]
    return options


# =========================
# Dataset Summary
# =========================
def format_summary(df: pd.DataFrame) -> Dict[str, object]:
    numeric_cols = numeric_columns(df)
    categorical_cols = categorical_columns(df)
    target_col = pick_target_column(df)
    missing_pct = (df.isna().sum().sum() / max(df.size, 1)) * 100

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_pct": round(float(missing_pct), 2),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(categorical_cols)),
        "target": target_col,
        "preview_columns": list(df.columns),
        "preview_rows": df.head(12).fillna("").to_dict(orient="records"),
    }


# =========================
# Dashboard Charts
# =========================
def create_target_balance_chart(y: pd.Series) -> str:
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    wedges, texts, autotexts = ax.pie(
        [positive, negative],
        startangle=90,
        counterclock=False,
        colors=["#5b8fff", "#4af0c4"],
        wedgeprops=dict(width=0.35, edgecolor="#0b1320"),
        autopct=lambda pct: f"{pct:.0f}%",
    )
    ax.set_title("Rainfall Class Balance")
    ax.legend(
        wedges,
        [f"Rain: {positive}", f"No Rain: {negative}"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    ax.set_aspect("equal")
    return fig_to_uri(fig)


def create_class_average_chart(df: pd.DataFrame, target_col: str, numeric_cols: List[str]) -> str:
    usable = numeric_cols[:6]
    if not usable:
        return ""
    positives = df[df[target_col].astype(str).str.strip().str.lower().isin(POSITIVE_LABELS)]
    negatives = df[df[target_col].astype(str).str.strip().str.lower().isin(NEGATIVE_LABELS)]
    means = pd.DataFrame(
        {
            "Rain": positives[usable].mean(numeric_only=True),
            "No Rain": negatives[usable].mean(numeric_only=True),
        }
    ).fillna(0)
    plot_df = means.reset_index().rename(columns={"index": "Feature"})
    plot_df = plot_df.melt(id_vars="Feature", var_name="Class", value_name="Mean")
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    sns.barplot(data=plot_df, x="Feature", y="Mean", hue="Class", ax=ax, palette=["#5b8fff", "#4af0c4"])
    ax.set_title("Feature Averages by Rainfall Class")
    ax.set_xlabel("")
    ax.set_ylabel("Mean")
    ax.tick_params(axis="x", rotation=25)
    return fig_to_uri(fig)


def create_distribution_chart(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    cols = numeric_cols[:4]
    if not cols:
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for axis, column in zip(axes, cols):
        sns.histplot(df[column].dropna(), bins=25, kde=True, ax=axis, color="#71d6ff")
        axis.set_title(column)
        axis.set_xlabel("")
        axis.set_ylabel("Count")
    for axis in axes[len(cols):]:
        axis.axis("off")
    fig.suptitle("Numeric Feature Distributions", y=1.02)
    return fig_to_uri(fig)


def create_correlation_chart(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    if len(numeric_cols) < 2:
        return ""
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=True)
    ax.set_title("Correlation Heatmap")
    return fig_to_uri(fig)


def create_category_chart(df: pd.DataFrame, categorical_cols: List[str]) -> str:
    if not categorical_cols:
        return ""
    column = categorical_cols[0]
    counts = df[column].astype(str).value_counts().head(8).reset_index()
    counts.columns = [column, "Count"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.barplot(data=counts, x=column, y="Count", ax=ax, color="#4af0c0")
    ax.set_title(f"Category Breakdown - {column}")
    ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("")
    return fig_to_uri(fig)


def create_model_comparison_chart(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return ""
    plot_df = summary_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "CV AUC"], var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Score Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)
    return fig_to_uri(fig)


def create_confusion_matrix_chart(cm: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", ax=ax, cbar=True)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig_to_uri(fig)


def create_roc_chart(y_true: pd.Series, probabilities: np.ndarray, auc_value: float, model_name: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(fpr, tpr, color="#71d6ff", linewidth=2.5, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8da2b7", label="Random")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig_to_uri(fig)


def create_feature_importance_chart(model_name: str, model, feature_names: List[str]) -> str:
    importance_values = None
    if hasattr(model, "named_steps"):
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            importance_values = np.abs(classifier.coef_[0])
        elif hasattr(classifier, "feature_importances_"):
            importance_values = classifier.feature_importances_

    if importance_values is None or not feature_names:
        return ""

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names[: len(importance_values)],
            "Importance": importance_values[: len(feature_names)],
        }
    ).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax, color="#4af0c0")
    ax.set_title(f"Top Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    return fig_to_uri(fig)


def create_prediction_form(feature_columns: List[str], numeric_cols: List[str], defaults: Dict[str, object], options: Dict[str, List[str]]) -> List[Dict[str, object]]:
    fields: List[Dict[str, object]] = []
    for column in feature_columns:
        if column in numeric_cols:
            fields.append({"name": column, "kind": "number", "default": float(defaults.get(column, 0.0) or 0.0)})
        else:
            field_options = options.get(column) or [str(defaults.get(column, "Unknown"))]
            fields.append(
                {
                    "name": column,
                    "kind": "select",
                    "default": str(defaults.get(column, field_options[0])),
                    "options": field_options,
                }
            )
    return fields


# =========================
# Training Workflow
# =========================
def train_models(
    feature_df: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    selected_model_names: List[str],
    test_size: float,
    random_state: int,
    cv_folds: int,
    enable_tuning: bool,
) -> Dict[str, object]:
    models = build_models()
    model_grids = build_model_grids()

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model_results = []
    fitted_models = {}

    for model_name in selected_model_names:
        base_pipeline = make_pipeline(models[model_name], numeric_cols, categorical_cols)
        estimator = base_pipeline
        cv_auc = None
        best_params = {}

        if enable_tuning and model_name in model_grids:
            # Try a few settings and keep the one that scores best on validation.
            search = GridSearchCV(
                base_pipeline,
                model_grids[model_name],
                scoring="roc_auc",
                cv=cv_folds,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params = search.best_params_
            cv_auc = float(search.best_score_)
            cv_scores = cross_val_score(estimator, X_train, y_train, scoring="roc_auc", cv=cv_folds, n_jobs=-1)
            cv_auc = float(cv_scores.mean())
        else:
            estimator.fit(X_train, y_train)
            cv_auc = None

        fitted_models[model_name] = estimator

        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
        auc = safe_metric_score(roc_auc_score, y_test, probabilities) if probabilities is not None else None
        model_results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "AUC": auc,
                "CV AUC": cv_auc,
                "Best Params": best_params,
                "Predictions": predictions,
                "Probabilities": probabilities,
            }
        )

    summary_df = pd.DataFrame(model_results).drop(columns=["Predictions", "Probabilities"])
    summary_df = summary_df.sort_values(["AUC", "Accuracy"], ascending=False, na_position="last").reset_index(drop=True)

    best_row = summary_df.iloc[0]
    best_model_name = str(best_row["Model"])
    best_model = fitted_models[best_model_name]
    best_result = next(result for result in model_results if result["Model"] == best_model_name)

    feature_defaults = build_feature_defaults(feature_df, numeric_cols, categorical_cols)
    category_options = build_category_options(feature_df, categorical_cols)
    feature_names = encode_feature_names(best_model.named_steps["preprocessor"], numeric_cols, categorical_cols)

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": list(feature_df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_defaults": feature_defaults,
        "category_options": category_options,
        "feature_names": feature_names,
    }

    global LATEST_ARTIFACT
    LATEST_ARTIFACT = artifact

    return {
        "summary": summary_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_result": best_result,
        "artifact": artifact,
        "X_test": X_test,
        "y_test": y_test,
        "fitted_models": fitted_models,
    }


# Run the full analysis workflow: detect the target, train models, and prepare results.
def analyze_dataset(df: pd.DataFrame, test_size: float, random_state: int, cv_folds: int, enable_tuning: bool, selected_models: List[str]) -> Dict[str, object]:
    target_col = pick_target_column(df)
    if target_col is None:
        raise ValueError("No target column found. Add a Rainfall/Target/Label column.")

    y, valid_index = parse_target_labels(df[target_col])
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes such as yes/no.")

    feature_df = df.loc[valid_index].drop(columns=[target_col]).copy()
    y = y.loc[valid_index]

    if feature_df.empty:
        raise ValueError("No usable rows remain after cleaning the target column.")

    numeric_cols = numeric_columns(feature_df)
    categorical_cols = categorical_columns(feature_df)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns were found for training.")

    selected_models = selected_models or list(build_models().keys())

    training = train_models(
        feature_df=feature_df,
        y=y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        selected_model_names=selected_models,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        enable_tuning=enable_tuning,
    )

    summary_df = training["summary"]
    best_model_name = training["best_model_name"]
    best_model = training["best_model"]
    best_result = training["best_result"]
    artifact = training["artifact"]
    X_test = training["X_test"]
    y_test = training["y_test"]

    best_predictions = best_result["Predictions"]
    best_probabilities = best_result["Probabilities"]
    auc_value = float(best_result["AUC"]) if best_result["AUC"] is not None else None

    # Prepare every chart the frontend may want to display.
    charts = [
        {"id": "target-balance", "title": "Rainfall Class Balance", "src": create_target_balance_chart(y)},
        {"id": "class-averages", "title": "Feature Averages by Rainfall Class", "src": create_class_average_chart(feature_df.join(y.rename(target_col)), target_col, numeric_cols)},
        {"id": "distributions", "title": "Numeric Feature Distributions", "src": create_distribution_chart(feature_df, numeric_cols)},
        {"id": "correlation", "title": "Correlation Heatmap", "src": create_correlation_chart(feature_df, numeric_cols)},
        {"id": "category", "title": "Category Breakdown", "src": create_category_chart(feature_df, categorical_cols)},
        {"id": "model-comparison", "title": "Model Score Comparison", "src": create_model_comparison_chart(summary_df)},
        {"id": "roc", "title": "ROC Curve", "src": create_roc_chart(y_test, best_probabilities, auc_value or 0.0, best_model_name) if best_probabilities is not None else ""},
        {"id": "confusion", "title": "Confusion Matrix", "src": create_confusion_matrix_chart(confusion_matrix(y_test, best_predictions))},
        {"id": "importance", "title": "Feature Importance", "src": create_feature_importance_chart(best_model_name, best_model, artifact["feature_names"])},
    ]
    charts = [chart for chart in charts if chart["src"]]

    report = classification_report(y_test, best_predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Label"})

    preview_table = feature_df.head(12).fillna("").to_dict(orient="records")
    prediction_fields = create_prediction_form(artifact["feature_columns"], artifact["numeric_cols"], artifact["feature_defaults"], artifact["category_options"])

    artifact_bytes = pickle.dumps(artifact)

    return {
        "summary": format_summary(df),
        "analysis": {
            "target_col": target_col,
            "best_model_name": best_model_name,
            "best_metrics": {
                "accuracy": float(best_result["Accuracy"]),
                "precision": float(best_result["Precision"]),
                "recall": float(best_result["Recall"]),
                "f1": float(best_result["F1"]),
                "auc": float(best_result["AUC"]) if best_result["AUC"] is not None else None,
                "cv_auc": float(best_result["CV AUC"]) if best_result["CV AUC"] is not None else None,
            },
            "model_summary": summary_df.fillna("").to_dict(orient="records"),
            "best_predictions": pd.DataFrame(
                {
                    "Actual": y_test.reset_index(drop=True),
                    "Predicted": best_predictions,
                    "Probability": best_probabilities if best_probabilities is not None else np.nan,
                }
            ).head(20).fillna("").to_dict(orient="records"),
            "classification_report": report_df.fillna("").to_dict(orient="records"),
            "prediction_fields": prediction_fields,
            "feature_columns": artifact["feature_columns"],
        },
        "preview": {
            "columns": list(feature_df.columns),
            "rows": preview_table,
        },
        "charts": charts,
        "download_name": "aether_best_model.pkl",
        "download_ready": True,
        "artifact_token": base64.b64encode(artifact_bytes).decode("utf-8"),
    }


@app.route("/")
def index():
    # Serve the static HTML dashboard page.
    return send_file(os.path.join(app.root_path, "demo.html"))


@app.route("/analyze", methods=["POST"])
def analyze():
    # Receive the uploaded dataset, run the analysis, and return JSON results.
    uploaded_file = request.files.get("dataset")
    if not uploaded_file:
        return jsonify({"error": "Please upload a CSV or Excel file."}), 400

    try:
        dataframe = normalize_frame(load_dataframe(uploaded_file))
        test_size = float(request.form.get("test_size", 0.25))
        random_state = int(request.form.get("random_state", 42))
        cv_folds = int(request.form.get("cv_folds", 5))
        enable_tuning = request.form.get("enable_tuning", "true").lower() == "true"
        selected_models = request.form.getlist("models") or list(build_models().keys())
        payload = analyze_dataset(dataframe, test_size, random_state, cv_folds, enable_tuning, selected_models)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    # Use the most recently trained model to make a single prediction.
    if not LATEST_ARTIFACT:
        return jsonify({"error": "Train a model first by uploading a dataset."}), 400

    payload = request.get_json(force=True, silent=True) or {}
    features = payload.get("features", {})
    artifact = LATEST_ARTIFACT
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    numeric_cols = artifact["numeric_cols"]
    default_values = artifact["feature_defaults"]

    row = {}
    for column in feature_columns:
        value = features.get(column, default_values.get(column))
        if column in numeric_cols:
            try:
                row[column] = float(value)
            except (TypeError, ValueError):
                row[column] = float(default_values.get(column, 0.0) or 0.0)
        else:
            row[column] = str(value)

    sample = pd.DataFrame([row], columns=feature_columns)
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1]) if hasattr(model, "predict_proba") else None
    return jsonify(
        {
            "prediction": prediction,
            "label": "Rain likely" if prediction == 1 else "No rain likely",
            "probability": probability,
        }
    )


@app.route("/download-model")
def download_model():
    # Let the user download the trained model for later reuse.
    if not LATEST_ARTIFACT:
        return jsonify({"error": "No trained model available."}), 400

    buffer = io.BytesIO()
    pickle.dump(LATEST_ARTIFACT, buffer)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="aether_best_model.pkl",
        mimetype="application/octet-stream",
    )


if __name__ == "__main__":
    app.run(debug=True)
