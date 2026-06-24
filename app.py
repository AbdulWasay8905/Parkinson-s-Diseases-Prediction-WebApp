from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except ImportError:  # pragma: no cover - pickle remains available as a fallback.
    joblib = None


APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "parkinsons_model.pkl"
SCALER_FILENAME = "scaler.pkl"

FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

FEATURE_TABS = {
    "Frequency Features": [
        "MDVP:Fo(Hz)",
        "MDVP:Fhi(Hz)",
        "MDVP:Flo(Hz)",
        "HNR",
        "NHR",
    ],
    "Jitter & Shimmer Features": [
        "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)",
        "MDVP:RAP",
        "MDVP:PPQ",
        "Jitter:DDP",
        "MDVP:Shimmer",
        "MDVP:Shimmer(dB)",
        "Shimmer:APQ3",
        "Shimmer:APQ5",
        "MDVP:APQ",
        "Shimmer:DDA",
    ],
    "Advanced Features": [
        "RPDE",
        "DFA",
        "spread1",
        "spread2",
        "D2",
        "PPE",
    ],
}

FEATURE_INPUTS = {
    "MDVP:Fo(Hz)": {"value": 120.0, "step": 0.001, "format": "%.5f"},
    "MDVP:Fhi(Hz)": {"value": 150.0, "step": 0.001, "format": "%.5f"},
    "MDVP:Flo(Hz)": {"value": 100.0, "step": 0.001, "format": "%.5f"},
    "MDVP:Jitter(%)": {"value": 0.006, "step": 0.0001, "format": "%.5f"},
    "MDVP:Jitter(Abs)": {"value": 0.00005, "step": 0.00001, "format": "%.5f"},
    "MDVP:RAP": {"value": 0.003, "step": 0.0001, "format": "%.5f"},
    "MDVP:PPQ": {"value": 0.003, "step": 0.0001, "format": "%.5f"},
    "Jitter:DDP": {"value": 0.009, "step": 0.0001, "format": "%.5f"},
    "MDVP:Shimmer": {"value": 0.030, "step": 0.0001, "format": "%.5f"},
    "MDVP:Shimmer(dB)": {"value": 0.300, "step": 0.001, "format": "%.5f"},
    "Shimmer:APQ3": {"value": 0.015, "step": 0.0001, "format": "%.5f"},
    "Shimmer:APQ5": {"value": 0.020, "step": 0.0001, "format": "%.5f"},
    "MDVP:APQ": {"value": 0.025, "step": 0.0001, "format": "%.5f"},
    "Shimmer:DDA": {"value": 0.045, "step": 0.0001, "format": "%.5f"},
    "NHR": {"value": 0.020, "step": 0.0001, "format": "%.5f"},
    "HNR": {"value": 20.0, "step": 0.001, "format": "%.5f"},
    "RPDE": {"value": 0.500, "step": 0.001, "format": "%.5f"},
    "DFA": {"value": 0.700, "step": 0.001, "format": "%.5f"},
    "spread1": {"value": -5.000, "step": 0.001, "format": "%.5f"},
    "spread2": {"value": 0.250, "step": 0.001, "format": "%.5f"},
    "D2": {"value": 2.300, "step": 0.001, "format": "%.5f"},
    "PPE": {"value": 0.200, "step": 0.001, "format": "%.5f"},
}

FEATURE_WIDGET_KEYS = {
    feature: f"feature_{index:02d}" for index, feature in enumerate(FEATURE_NAMES, start=1)
}


@dataclass(frozen=True)
class ModelAssets:
    model: Any | None
    scaler: Any | None
    model_path: Path | None
    scaler_path: Path | None
    model_error: str | None = None
    scaler_error: str | None = None
    model_warning: str | None = None


def configure_page() -> None:
    st.set_page_config(
        page_title="Parkinson's Disease Prediction System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def get_custom_css() -> str:
    return """
    <style>
        :root {
            --medical-navy: #12324a;
            --clinical-teal: #0c8f8f;
            --soft-mint: #e6f7f3;
            --pale-blue: #eef6ff;
            --ink: #1f2d3d;
            --muted: #64748b;
            --danger: #c62828;
            --danger-bg: #fff1f1;
            --success: #087f5b;
            --success-bg: #edfff8;
            --border: #dbe7ef;
            /* Force Light Theme Inputs */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: white !important;
    color: black !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: white !important;
    color: black !important;
}

.stNumberInput button {
    background: white !important;
    color: black !important;
}

.stSelectbox svg {
    color: black !important;
}

/* Labels */

label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label {
    color: #12324a !important;
    font-weight: 600;
}

/* Tabs */

button[role="tab"] {
    color: #12324a !important;
}

/* File uploader */

[data-testid="stFileUploader"] section {
    background: white !important;
}

/* Metrics */

[data-testid="stMetric"] {
    background: white !important;
}
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(238, 246, 255, 0.96), rgba(246, 252, 250, 0.98)),
                repeating-linear-gradient(45deg, rgba(12, 143, 143, 0.03) 0 1px, transparent 1px 18px);
            color: var(--ink);
            input,
textarea,
select {
    background-color: white !important;
    color: black !important;
}

.stTextInput input,
.stNumberInput input {
    background-color: white !important;
    color: black !important;
}
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12324a 0%, #0b4d5c 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f8fbff;
        }

        [data-testid="stSidebar"] .stAlert {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.20);
            color: #f8fbff;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        .app-header {
            border: 1px solid var(--border);
            background: linear-gradient(135deg, #ffffff 0%, #f4fbfb 46%, #eef6ff 100%);
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 16px 45px rgba(18, 50, 74, 0.08);
            margin-bottom: 1.25rem;
        }

        .app-header h1 {
            color: var(--medical-navy);
            font-size: 2.55rem;
            line-height: 1.15;
            margin: 0 0 0.5rem 0;
            letter-spacing: 0;
        }

        .app-header p {
            color: var(--muted);
            font-size: 1.1rem;
            margin: 0;
        }

        .section-title {
            color: var(--medical-navy);
            font-size: 1.35rem;
            font-weight: 700;
            margin: 1.25rem 0 0.65rem 0;
        }

        .section-kicker {
            color: var(--clinical-teal);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.74rem;
            margin-bottom: 0.2rem;
        }

        .info-band {
            border-left: 5px solid var(--clinical-teal);
            background: #ffffff;
            padding: 1rem 1.15rem;
            border-radius: 8px;
            border-top: 1px solid var(--border);
            border-right: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            margin: 1rem 0;
        }

        .result-card {
            border-radius: 8px;
            padding: 1.35rem 1.5rem;
            border: 1px solid var(--border);
            margin: 1rem 0;
        }

        .result-card h3 {
            margin: 0 0 0.45rem 0;
            font-size: 1.45rem;
            letter-spacing: 0;
        }

        .result-card p {
            margin: 0.2rem 0;
            color: var(--ink);
        }

        .success-card {
            background: var(--success-bg);
            border-color: rgba(8, 127, 91, 0.28);
        }

        .success-card h3 {
            color: var(--success);
        }

        .warning-card {
            background: var(--danger-bg);
            border-color: rgba(198, 40, 40, 0.28);
        }

        .warning-card h3 {
            color: var(--danger);
        }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0;
        }

        .sidebar-card h3 {
            font-size: 1rem;
            margin: 0 0 0.4rem 0;
        }

        .sidebar-card p,
        .sidebar-card li {
            color: #eaf4fb;
            font-size: 0.92rem;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 1.5rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(18, 50, 74, 0.05);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: var(--medical-navy);
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 8px;
            border: 1px solid #0c8f8f;
            background: #0c8f8f;
            color: #ffffff;
            font-weight: 700;
            min-height: 2.75rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: #096f70;
            background: #096f70;
            color: #ffffff;
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
        }

        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }

            .app-header {
                padding: 1.25rem;
            }

            .app-header h1 {
                font-size: 1.8rem;
            }

            .app-header p {
                font-size: 1rem;
            }
        }
    </style>
    """


def resolve_model_path() -> tuple[Path | None, str | None]:
    primary_path = APP_DIR / MODEL_FILENAME
    if primary_path.exists():
        return primary_path, None

    fallback_names = [
        "Parkionmson's_model.pkl",
        "parkinsons_model.joblib",
        "model.pkl",
    ]
    for filename in fallback_names:
        candidate = APP_DIR / filename
        if candidate.exists():
            warning = (
                f"Expected `{MODEL_FILENAME}` but loaded fallback model "
                f"`{candidate.name}`. Rename it before deploying for best portability."
            )
            return candidate, warning

    discovered_models = sorted(APP_DIR.glob("*model*.pkl"))
    if discovered_models:
        candidate = discovered_models[0]
        warning = (
            f"Expected `{MODEL_FILENAME}` but loaded discovered model "
            f"`{candidate.name}`."
        )
        return candidate, warning

    return None, None


def load_serialized_object(path: Path) -> Any:
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    with path.open("rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_model_assets() -> ModelAssets:
    model_path, model_warning = resolve_model_path()
    model = None
    model_error = None

    if model_path is None:
        model_error = (
            f"`{MODEL_FILENAME}` was not found. Place the trained model in the "
            "project root and restart the app."
        )
    else:
        try:
            model = load_serialized_object(model_path)
        except Exception as exc:
            model_error = f"Unable to load `{model_path.name}`: {exc}"

    scaler_path = APP_DIR / SCALER_FILENAME
    scaler = None
    scaler_error = None
    if scaler_path.exists():
        try:
            scaler = load_serialized_object(scaler_path)
        except Exception as exc:
            scaler_error = f"Unable to load `{SCALER_FILENAME}`: {exc}"
    else:
        scaler_path = None

    return ModelAssets(
        model=model,
        scaler=scaler,
        model_path=model_path,
        scaler_path=scaler_path,
        model_error=model_error,
        scaler_error=scaler_error,
        model_warning=model_warning,
    )


def reset_form_state() -> None:
    keys_to_clear = [
        "patient_name",
        "patient_age",
        "patient_gender",
        "last_prediction",
        "bulk_prediction_results",
    ]
    keys_to_clear.extend(FEATURE_WIDGET_KEYS.values())
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🧠 Parkinson's Disease Prediction System</h1>
            <p>AI-powered early screening using voice measurements.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(assets: ModelAssets) -> None:
    st.sidebar.markdown("## 🏥 Parkinson Screening")
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h3>📌 Project Description</h3>
            <p>
                This dashboard predicts whether voice measurements are more
                consistent with a healthy profile or Parkinson's Disease.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h3>👨‍💻 Developer</h3>
            <p>Abdul Wasay</p>
            <p>Machine Learning Project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_status = "Active" if assets.model is not None else "Unavailable"
    scaler_status = "Loaded" if assets.scaler is not None else "Not found"
    model_name = assets.model_path.name if assets.model_path else MODEL_FILENAME
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <h3>🧪 Model Information</h3>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Status:</strong> {model_status}</p>
            <p><strong>Scaler:</strong> {scaler_status}</p>
            <p><strong>Target:</strong> 0 = Healthy, 1 = Parkinson's Disease</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h3>📝 Instructions</h3>
            <ul>
                <li>Enter patient details.</li>
                <li>Provide all 22 voice measurements.</li>
                <li>Run prediction for one patient or upload a CSV for bulk screening.</li>
                <li>Use results as an early screening aid only.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if assets.model_warning:
        st.sidebar.warning(assets.model_warning)
    if assets.model_error:
        st.sidebar.error(assets.model_error)
    if assets.scaler_error:
        st.sidebar.warning(assets.scaler_error)


def render_status_alerts(assets: ModelAssets) -> None:
    if assets.model_error:
        st.error(assets.model_error)
    elif assets.model_warning:
        st.warning(assets.model_warning)

    if assets.scaler_error:
        st.warning(assets.scaler_error)
    elif assets.scaler is None:
        st.info("Optional `scaler.pkl` was not found. Predictions will use raw feature values.")


def feature_frame_from_values(feature_values: dict[str, float]) -> pd.DataFrame:
    ordered_values = [feature_values[feature] for feature in FEATURE_NAMES]
    return pd.DataFrame([ordered_values], columns=FEATURE_NAMES)


def apply_optional_scaler(feature_df: pd.DataFrame, scaler: Any | None) -> Any:
    if scaler is None:
        return feature_df

    try:
        return scaler.transform(feature_df)
    except Exception:
        return scaler.transform(feature_df.to_numpy())


def safe_predict(model: Any, prepared_features: Any) -> tuple[np.ndarray, Any]:
    try:
        predictions = model.predict(prepared_features)
        return np.asarray(predictions), prepared_features
    except Exception:
        array_features = (
            prepared_features.to_numpy()
            if isinstance(prepared_features, pd.DataFrame)
            else np.asarray(prepared_features)
        )
        predictions = model.predict(array_features)
        return np.asarray(predictions), array_features


def safe_predict_proba(model: Any, prepared_features: Any) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    try:
        return np.asarray(model.predict_proba(prepared_features))
    except Exception:
        try:
            array_features = (
                prepared_features.to_numpy()
                if isinstance(prepared_features, pd.DataFrame)
                else np.asarray(prepared_features)
            )
            return np.asarray(model.predict_proba(array_features))
        except Exception:
            return None


def confidence_for_predictions(
    model: Any,
    prepared_features: Any,
    predictions: np.ndarray,
) -> tuple[list[float | None], np.ndarray | None]:
    flat_predictions = np.asarray(predictions).ravel()
    probabilities = safe_predict_proba(model, prepared_features)
    if probabilities is None:
        return [None for _ in flat_predictions], None

    classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
    confidence_scores: list[float | None] = []

    for prediction, probability_row in zip(flat_predictions, probabilities):
        predicted_label = int(prediction)
        if predicted_label in classes:
            class_index = classes.index(predicted_label)
        else:
            class_index = int(np.argmax(probability_row))
        confidence_scores.append(round(float(probability_row[class_index]) * 100, 2))

    return confidence_scores, probabilities


def run_prediction(
    model: Any,
    scaler: Any | None,
    feature_df: pd.DataFrame,
) -> tuple[int, float | None, np.ndarray | None]:
    prepared_features = apply_optional_scaler(feature_df, scaler)
    predictions, prediction_features = safe_predict(model, prepared_features)
    confidence_scores, probabilities = confidence_for_predictions(
        model,
        prediction_features,
        predictions,
    )
    return int(predictions.ravel()[0]), confidence_scores[0], probabilities


def render_feature_inputs(features: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    columns = st.columns(2)

    for index, feature in enumerate(features):
        config = FEATURE_INPUTS[feature]
        with columns[index % 2]:
            values[feature] = st.number_input(
                label=feature,
                value=float(config["value"]),
                step=float(config["step"]),
                format=str(config["format"]),
                key=FEATURE_WIDGET_KEYS[feature],
            )

    return values


def render_prediction_form(assets: ModelAssets) -> None:
    top_left, top_right = st.columns([0.72, 0.28])
    with top_left:
        st.markdown('<div class="section-kicker">Single Patient Screening</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    with top_right:
        st.button("↺ Reset Form", use_container_width=True, on_click=reset_form_state)

    with st.form("prediction_form"):
        patient_col_1, patient_col_2, patient_col_3 = st.columns([2, 1, 1])
        with patient_col_1:
            patient_name = st.text_input("Patient Name", key="patient_name")
        with patient_col_2:
            patient_age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=45,
                step=1,
                key="patient_age",
            )
        with patient_col_3:
            patient_gender = st.selectbox(
                "Gender",
                ["Prefer not to say", "Female", "Male", "Other"],
                key="patient_gender",
            )

        st.markdown('<div class="section-title">Voice Feature Inputs</div>', unsafe_allow_html=True)
        tab_frequency, tab_jitter, tab_advanced = st.tabs(
            ["🎙️ Frequency Features", "〽️ Jitter & Shimmer Features", "🧬 Advanced Features"]
        )

        feature_values: dict[str, float] = {}
        with tab_frequency:
            feature_values.update(render_feature_inputs(FEATURE_TABS["Frequency Features"]))
        with tab_jitter:
            feature_values.update(render_feature_inputs(FEATURE_TABS["Jitter & Shimmer Features"]))
        with tab_advanced:
            feature_values.update(render_feature_inputs(FEATURE_TABS["Advanced Features"]))

        submitted = st.form_submit_button(
            "🔍 Run Prediction",
            use_container_width=True,
            disabled=assets.model is None,
        )

    if submitted and assets.model is not None:
        with st.spinner("Analyzing voice measurements..."):
            time.sleep(0.5)
            try:
                feature_df = feature_frame_from_values(feature_values)
                prediction, confidence, probabilities = run_prediction(
                    assets.model,
                    assets.scaler,
                    feature_df,
                )
                st.session_state["last_prediction"] = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "patient_name": patient_name.strip() or "Not provided",
                    "patient_age": int(patient_age),
                    "patient_gender": patient_gender,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


def confidence_text(confidence: float | None) -> str:
    return f"{confidence:.2f}%" if confidence is not None else "N/A"


def render_dashboard_metrics(assets: ModelAssets) -> None:
    last_prediction = st.session_state.get("last_prediction")
    confidence = None if last_prediction is None else last_prediction.get("confidence")
    confidence_value = "Pending" if last_prediction is None else confidence_text(confidence)

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Number of Features", len(FEATURE_NAMES))
    metric_col_2.metric("Model Status", "Active" if assets.model is not None else "Missing")
    metric_col_3.metric("Prediction Confidence", confidence_value)


def render_result_section() -> None:
    last_prediction = st.session_state.get("last_prediction")
    if not last_prediction:
        st.markdown(
            """
            <div class="info-band">
                <strong>Ready for screening.</strong>
                Enter the patient information and voice measurements, then run the prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction = int(last_prediction["prediction"])
    confidence = last_prediction.get("confidence")
    patient_name = last_prediction["patient_name"]

    if prediction == 1:
        card_class = "warning-card"
        title = "⚠️ Parkinson's Disease Detected"
        message = (
            "The model detected a Parkinson's-positive pattern in the supplied "
            "voice measurements. A qualified clinician should review the case."
        )
    else:
        card_class = "success-card"
        title = "✅ No Signs of Parkinson's Disease"
        message = (
            "The model detected a healthy pattern in the supplied voice measurements. "
            "Continue using clinical judgement for any symptoms or concerns."
        )

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <h3>{title}</h3>
            <p><strong>Patient:</strong> {patient_name}</p>
            <p><strong>Confidence:</strong> {confidence_text(confidence)}</p>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence is not None:
        confidence_col, diagnosis_col = st.columns(2)
        confidence_col.metric("Probability Confidence", f"{confidence:.2f}%")
        diagnosis_col.metric(
            "Predicted Class",
            "1 - Parkinson's Disease" if prediction == 1 else "0 - Healthy",
        )
    else:
        st.info("This model does not expose `predict_proba()`, so probability confidence is unavailable.")


def validate_bulk_dataframe(data: pd.DataFrame) -> list[str]:
    return [feature for feature in FEATURE_NAMES if feature not in data.columns]


def prepare_bulk_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        bad_columns = features.columns[features.isnull().any()].tolist()
        joined_columns = ", ".join(bad_columns)
        raise ValueError(f"CSV contains missing or non-numeric values in: {joined_columns}")
    return features


def run_bulk_predictions(model: Any, scaler: Any | None, data: pd.DataFrame) -> pd.DataFrame:
    feature_df = prepare_bulk_features(data)
    prepared_features = apply_optional_scaler(feature_df, scaler)
    predictions, prediction_features = safe_predict(model, prepared_features)
    predicted_labels = predictions.astype(int).ravel()
    confidence_scores, _ = confidence_for_predictions(model, prediction_features, predictions)

    results = data.copy()
    results["prediction"] = predicted_labels
    results["prediction_label"] = np.where(
        predicted_labels == 1,
        "Parkinson's Disease",
        "Healthy",
    )
    results["confidence_percent"] = [
        score if score is not None else "N/A" for score in confidence_scores
    ]
    return results


def render_bulk_prediction_section(assets: ModelAssets) -> None:
    st.markdown('<div class="section-title">Bulk CSV Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-band">
            Upload a CSV containing the 22 training features. Extra columns are allowed and will be preserved.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload patient voice measurement CSV",
        type=["csv"],
        disabled=assets.model is None,
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            st.caption(f"Uploaded {len(uploaded_data):,} row(s) and {len(uploaded_data.columns):,} column(s).")

            missing_features = validate_bulk_dataframe(uploaded_data)
            if missing_features:
                st.error(
                    "CSV is missing required feature columns: "
                    + ", ".join(missing_features)
                )
                return

            if st.button("📊 Run Bulk Prediction", use_container_width=True):
                with st.spinner("Running bulk predictions..."):
                    try:
                        st.session_state["bulk_prediction_results"] = run_bulk_predictions(
                            assets.model,
                            assets.scaler,
                            uploaded_data,
                        )
                    except Exception as exc:
                        st.error(f"Bulk prediction failed: {exc}")
        except Exception as exc:
            st.error(f"Unable to read uploaded CSV: {exc}")

    bulk_results = st.session_state.get("bulk_prediction_results")
    if isinstance(bulk_results, pd.DataFrame):
        st.dataframe(bulk_results, use_container_width=True)
        csv_data = bulk_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Prediction Results",
            data=csv_data,
            file_name=f"parkinsons_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <p class="footer-note">
            Medical disclaimer: this application is an educational screening aid and is not a clinical diagnosis.
            Always consult a qualified healthcare professional for medical decisions.
        </p>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    assets = load_model_assets()

    render_sidebar(assets)
    render_header()
    render_status_alerts(assets)
    render_prediction_form(assets)
    render_dashboard_metrics(assets)
    render_result_section()
    render_bulk_prediction_section(assets)
    render_footer()


if __name__ == "__main__":
    main()
