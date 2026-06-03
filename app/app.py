import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroRendement • Burundi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Styling (refonte CSS)
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700;900&family=Fraunces:wght@600;700&display=swap');

html, body { font-family: Inter, sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 10%, #1b3b6a 0%, rgba(27,59,106,0) 45%),
                radial-gradient(circle at 85% 35%, #0f766e 0%, rgba(15,118,110,0) 40%),
                linear-gradient(180deg, #071427 0%, #0b1f33 55%, #071427 100%);
    color: #EAF2FF;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,20,40,0.95) 0%, rgba(15,36,70,0.95) 100%);
    border-right: 1px solid rgba(96,165,250,0.25);
}
[data-testid="stSidebar"] * { color: #D6E4FF !important; }

.app-brand {
    font-family: Fraunces, serif;
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: -0.8px;
    background: linear-gradient(90deg, #60a5fa 0%, #34d399 45%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin: 14px 0 6px;
}

.app-tagline {
    text-align: center;
    color: rgba(214,228,255,0.8);
    font-size: 0.95rem;
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.kpi {
    display:flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px;
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(147,197,253,0.14);
}
.kpi .v { font-size: 2.1rem; font-weight: 900; letter-spacing: -0.5px; }
.kpi .l { color: rgba(214,228,255,0.7); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1px; }

.stButton > button {
    background: linear-gradient(135deg, rgba(96,165,250,1) 0%, rgba(52,211,153,1) 55%, rgba(251,191,36,1) 100%) !important;
    color: #051225 !important;
    font-weight: 900 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.8rem !important;
    box-shadow: 0 10px 24px rgba(0,0,0,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(96,165,250,0.2) !important;
    color: #EAF2FF !important;
    border-radius: 10px !important;
}

label { color: rgba(214,228,255,0.95) !important; }

.banner {
    border-radius: 16px;
    padding: 18px;
    border: 1px solid rgba(147,197,253,0.22);
}
.banner.good { background: linear-gradient(180deg, rgba(16,185,129,0.20) 0%, rgba(16,185,129,0.07) 100%); border-color: rgba(16,185,129,0.35); }
.banner.bad { background: linear-gradient(180deg, rgba(239,68,68,0.18) 0%, rgba(239,68,68,0.06) 100%); border-color: rgba(239,68,68,0.34); }
.banner .big { font-size: 2.0rem; font-weight: 950; margin: 0 0 6px; }
.banner .sub { color: rgba(234,242,255,0.85); margin: 0; }
.banner .proba { font-size: 2.4rem; font-weight: 950; margin: 10px 0 0; }

.divider { height: 1px; background: rgba(96,165,250,0.16); margin: 14px 0; }

.footer { margin-top: 26px; padding-top: 10px; text-align: center; color: rgba(214,228,255,0.55); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Load models & metadata
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))

    dt = joblib.load(os.path.join(models_dir, "decision_tree.pkl"))
    rf = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    lr = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    with open(os.path.join(models_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return dt, rf, lr, scaler, meta


dt_model, rf_model, lr_model, scaler, meta = load_artifacts()

FEATURE_COLS = meta["feature_cols"]
NUM_COLS = meta["num_cols"]
METRICS = meta["metrics"]

PROVINCES = [
    "Bubanza",
    "Bujumbura Rural",
    "Bururi",
    "Cankuzo",
    "Cibitoke",
    "Gitega",
    "Kayanza",
    "Kirundo",
    "Makamba",
    "Muramvya",
    "Muyinga",
    "Mwaro",
    "Ngozi",
    "Rutana",
    "Ruyigi",
]

CULTURES = ["Maïs", "Haricot", "Manioc", "Patate douce", "Sorgho", "Bananier"]

MODEL_CHOICES = {
    "🌲 Decision Tree": ("decision_tree", dt_model),
    "🌳 Random Forest": ("random_forest", rf_model),
    "📈 Régression Logistique": ("logistic_regression", lr_model),
}

# ─────────────────────────────────────────────────────────────
# Inference helpers (refonte)
# ─────────────────────────────────────────────────────────────

def _blank_row():
    return {c: 0 for c in FEATURE_COLS}


def _fill_numeric(row, altitude, pluv, temp, superficie, engrais, irrigation):
    row["altitude_m"] = altitude
    row["pluviometrie_mm"] = pluv
    row["temperature_moy_C"] = temp
    row["superficie_ha"] = superficie
    row["utilisation_engrais"] = engrais
    row["acces_irrigation"] = irrigation


def _fill_onehot(row, province, culture, saison_code):
    if saison_code == "B":
        sb = "saison_B"
        if sb in row:
            row[sb] = 1

    pc = f"province_{province}"
    if pc in row:
        row[pc] = 1

    cc = f"culture_{culture}"
    if cc in row:
        row[cc] = 1


def build_input(province, culture, saison_code, altitude, pluv, temp, superficie, engrais, irrigation):
    row = _blank_row()
    _fill_numeric(row, altitude, pluv, temp, superficie, engrais, irrigation)
    _fill_onehot(row, province, culture, saison_code)

    X = pd.DataFrame([row], columns=FEATURE_COLS)
    X[NUM_COLS] = scaler.transform(X[NUM_COLS])
    return X


def _proba_class_one(model, X):
    proba_vec = model.predict_proba(X)[0]
    if hasattr(model, "classes_") and 1 in list(model.classes_):
        idx = list(model.classes_).index(1)
    else:
        idx = 1 if len(proba_vec) > 1 else 0
    return float(proba_vec[idx])


def category_from(p1: float) -> str:
    if p1 >= 0.70:
        return "Risque faible (tendance favorable)"
    if p1 >= 0.50:
        return "Zone neutre (à surveiller)"
    return "Risque élevé (tendance défavorable)"


def make_semicircle_gauge(p1: float):
    color = "#22c55e" if p1 >= 0.5 else "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p1 * 100,
            number={"suffix": "%", "font": {"size": 28, "color": color}},
            title={"text": "Probabilité", "font": {"size": 13, "color": "#93C5FD"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.22},
                "bgcolor": "rgba(255,255,255,0.02)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(239,68,68,0.12)"},
                    {"range": [33, 66], "color": "rgba(250,204,21,0.12)"},
                    {"range": [66, 100], "color": "rgba(34,197,94,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF2FF"),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='app-brand'>AgroRendement Burundi</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-tagline'>Simulation de probabilité de bonne récolte</div>", unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### Paramètres de parcelle")
    st.caption("Entrées → encodage one-hot + normalisation → probabilité")
    st.markdown("---")

    province = st.selectbox("Province", PROVINCES, index=PROVINCES.index("Kayanza"))
    culture = st.selectbox("Culture", CULTURES, index=0)
    saison_label = st.selectbox("Saison", ["Mars–Juin (A)", "Sept–Déc (B)"])
    saison_code = "A" if saison_label.startswith("Mars") else "B"

    st.markdown("---")

    altitude = st.slider("Altitude (m)", 400, 2600, 1720, step=10)
    pluv = st.slider("Pluviométrie (mm)", 150, 1600, 900, step=10)
    temp = st.slider("Température (°C)", 14.0, 30.0, 18.2, step=0.1)
    superficie = st.slider("Superficie (ha)", 5.0, 800.0, 50.0, step=5.0)

    st.markdown("---")

    engrais = st.checkbox("Utilisation d'engrais", value=True)
    irrigation = st.checkbox("Accès à l'irrigation", value=False)

    st.markdown("---")

    model_label = st.selectbox("Modèle", list(MODEL_CHOICES.keys()), index=0)
    run = st.button("Lancer l’estimation", width="stretch")


if run:
    model_key, model_obj = MODEL_CHOICES[model_label]

    X_user = build_input(
        province=province,
        culture=culture,
        saison_code=saison_code,
        altitude=altitude,
        pluv=pluv,
        temp=temp,
        superficie=superficie,
        engrais=int(engrais),
        irrigation=int(irrigation),
    )

    p1 = _proba_class_one(model_obj, X_user)

    banner_class = "good" if p1 >= 0.5 else "bad"
    banner_title = "Prévision favorable" if p1 >= 0.5 else "Prévision défavorable"
    proba_color = "#22c55e" if p1 >= 0.5 else "#ef4444"

    c_left, c_right = st.columns([2.2, 1.0])

    with c_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='banner {banner_class}'>
                <p class='big'>{banner_title}</p>
                <p class='sub'>{province} — {culture}</p>
                <p class='proba' style='color:{proba_color}'>{p1*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.success(category_from(p1))

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        tab_metrics = st.tabs(["Métriques"])[0]
        with tab_metrics:
            m = METRICS[model_key]
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(
                    f"""
                    <div class='kpi'>
                        <div class='v'>{m['accuracy']:.1%}</div>
                        <div class='l'>Accuracy</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
                    <div class='kpi'>
                        <div class='v'>{m['f1']:.1%}</div>
                        <div class='l'>F1</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f"""
                    <div class='kpi'>
                        <div class='v'>{m['auc']:.3f}</div>
                        <div class='l'>AUC</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(make_semicircle_gauge(p1), width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class='footer'>
    Interface Streamlit • DT / RF / LR
</div>
""",
    unsafe_allow_html=True,
)

