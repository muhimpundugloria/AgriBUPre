

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os

# ─── Configuration page ───────────────────────────────────────
st.set_page_config(
    page_title="AgriPredict Burundi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&family=Playfair+Display:wght@700&display=swap');

/* Reset & base */
html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

/* Background principal */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #e8f4f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #112240 100%);
    border-right: 1px solid #1e4d6b;
}
[data-testid="stSidebar"] * { color: #ccd6f6 !important; }

/* Titre principal */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #64ffda, #00b4d8, #48cae4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.main-subtitle {
    text-align: center;
    color: #8892b0;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Cards métriques */
.metric-card {
    background: rgba(17, 34, 64, 0.85);
    border: 1px solid rgba(100, 255, 218, 0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(100, 255, 218, 0.4);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 900;
    color: #64ffda;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}

/* Prediction box */
.pred-good {
    background: linear-gradient(135deg, rgba(0, 200, 83, 0.2), rgba(0, 230, 118, 0.1));
    border: 2px solid #00c853;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.pred-bad {
    background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(229, 57, 53, 0.1));
    border: 2px solid #f44336;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.pred-title { font-size: 1.8rem; font-weight: 900; margin: 0; }
.pred-subtitle { font-size: 0.9rem; color: #aaa; margin-top: 0.5rem; }
.pred-proba { font-size: 3rem; font-weight: 900; margin: 1rem 0 0; }

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #64ffda;
    border-bottom: 1px solid rgba(100, 255, 218, 0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #8892b0 !important;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #64ffda !important;
    border-bottom: 2px solid #64ffda !important;
}

/* Inputs & selects */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] {
    background: rgba(17, 34, 64, 0.8) !important;
    border-color: rgba(100, 255, 218, 0.2) !important;
    color: #ccd6f6 !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #64ffda, #00b4d8) !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(100, 255, 218, 0.3) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(100, 255, 218, 0.5) !important;
}

/* Bouton 'PRÉDIRE LA RÉCOLTE' (dernier bouton dans la sidebar) */
[data-testid="stSidebar"] .stButton > button:last-of-type {
    background: linear-gradient(135deg, #ffb74d, #ff8a65) !important;
    color: #0a1628 !important;
    font-weight: 900 !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35) !important;
}

/* Info boxes */
.info-box {
    background: rgba(100, 255, 218, 0.08);
    border-left: 3px solid #64ffda;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #ccd6f6;
}
.warning-box {
    background: rgba(255, 193, 7, 0.08);
    border-left: 3px solid #ffc107;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #ccd6f6;
}

/* Tables */
.dataframe { background: rgba(17, 34, 64, 0.6) !important; }

/* Scenario cards */
.scenario-card {
    background: rgba(17, 34, 64, 0.7);
    border: 1px solid rgba(100, 255, 218, 0.1);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #495670;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(100, 255, 218, 0.08);
}
</style>
""", unsafe_allow_html=True)

# ─── Chargement des modèles ───────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    models_dir = os.path.abspath(os.path.join(base, "..", "models"))
    
    dt     = joblib.load(os.path.join(models_dir, "decision_tree.pkl"))
    rf     = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    lr     = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    
    with open(os.path.join(models_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    return dt, rf, lr, scaler, meta

dt_model, rf_model, lr_model, scaler, meta = load_models()

FEATURE_COLS = meta["feature_cols"]
NUM_COLS     = meta["num_cols"]
METRICS      = meta["metrics"]
PROVINCES    = [
    "Bubanza", "Bujumbura Rural", "Bururi", "Cankuzo", "Cibitoke", "Gitega",
    "Kayanza", "Kirundo", "Makamba", "Muramvya", "Muyinga", "Mwaro",
    "Ngozi", "Rutana", "Ruyigi"
]
CULTURES     = ["Maïs", "Haricot", "Manioc", "Patate douce", "Sorgho", "Bananier"]

MODEL_MAP = {
    "🌳 Arbre de Décision":      ("decision_tree", dt_model),
    "🌲 Forêt Aléatoire":        ("random_forest", rf_model),
    "📈 Régression Logistique":  ("logistic_regression", lr_model),
}

# ─── Helpers ──────────────────────────────────────────────────
def encode_input(province, culture, saison_code, altitude, pluv, temp, superficie,
                 engrais, irrigation): 
    """Encode une entrée utilisateur selon le pipeline d'entraînement."""
    row = {col: 0 for col in FEATURE_COLS}
    
    # Variables numériques
    row["altitude_m"]        = altitude
    row["pluviometrie_mm"]   = pluv
    row["temperature_moy_C"] = temp
    row["superficie_ha"]     = superficie
    row["utilisation_engrais"] = engrais
    row["acces_irrigation"]    = irrigation
    
    # Encodage saison (drop_first → saison_B)
    if saison_code == "B":
        k = "saison_B"
        if k in row: row[k] = 1
    
    # Province (référence = Bujumbura)
    prov_col = f"province_{province}"
    if prov_col in row: row[prov_col] = 1
    
    # Culture (référence = Bananier)
    cult_col = f"culture_{culture}"
    if cult_col in row: row[cult_col] = 1
    
    X = pd.DataFrame([row])[FEATURE_COLS]
    X[NUM_COLS] = scaler.transform(X[NUM_COLS])
    return X

def predict_all(province, culture, saison, altitude, pluv, temp,
                superficie, engrais, irrigation):
    X = encode_input(province, culture, saison, altitude, pluv, temp,
                     superficie, engrais, irrigation)
    results = {}
    for label, (key, model) in MODEL_MAP.items():
        proba_values = model.predict_proba(X)[0]
        # Find the probability for class 1 if labels are ordered differently
        class_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
        proba = float(proba_values[class_idx])
        pred = int(proba >= 0.5)
        results[label] = {"key": key, "pred": pred, "proba": proba}
    return results

def make_gauge(proba, title):
    color = "#00c853" if proba >= 0.5 else "#f44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#8892b0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#495670",
                     "tickfont": {"color": "#495670"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(17,34,64,0.5)",
            "bordercolor": "rgba(100,255,218,0.1)",
            "steps": [
                {"range": [0, 40],   "color": "rgba(244,67,54,0.12)"},
                {"range": [40, 60],  "color": "rgba(255,193,7,0.12)"},
                {"range": [60, 100], "color": "rgba(0,200,83,0.12)"},
            ],
            "threshold": {"line": {"color": "#64ffda", "width": 2},
                          "thickness": 0.8, "value": 50}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccd6f6"}
    )
    return fig

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,34,64,0.5)",
    font=dict(color="#ccd6f6", family="Lato"),
    xaxis=dict(gridcolor="rgba(100,255,218,0.06)", linecolor="rgba(100,255,218,0.15)"),
    yaxis=dict(gridcolor="rgba(100,255,218,0.06)", linecolor="rgba(100,255,218,0.15)"),
    margin=dict(t=40, b=40, l=40, r=20),
)

# ─── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🌾 AgriPredict Burundi</div>', unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres de la Parcelle")
    st.markdown("---")
    
    province  = st.selectbox("🗺️ Province",  PROVINCES, index=PROVINCES.index("Kayanza"))
    culture   = st.selectbox("🌱 Culture",   CULTURES,  index=0)
    # Saison: remettre le sélecteur pour permettre l'influence sur la prédiction
    saison = st.selectbox("📅 Saison", ["A (mars–juin)", "B (sept–déc)"], index=0)
    saison_code = "A" if saison.startswith("A") else "B"
    
    st.markdown("---")
    altitude  = st.slider("⛰️ Altitude (m)",      400, 2600, 1720, step=10)
    pluv      = st.slider("🌧️ Pluviométrie (mm)", 150, 1600, 900,  step=10)
    temp      = st.slider("🌡️ Température (°C)",  14.0, 30.0, 18.2, step=0.1)
    superficie= st.slider("📐 Superficie (ha)",   5.0, 800.0, 50.0, step=5.0)
    
    st.markdown("---")
    engrais   = st.checkbox("🧪 Utilisation d'engrais",  value=True)
    irrigation= st.checkbox("💧 Accès à l'irrigation",   value=False)
    
    st.markdown("---")
    model_choice = st.selectbox(
        "🤖 Modèle",
        list(MODEL_MAP.keys()),
        index=1
    )
    
    predict_btn = st.button("🔍  PRÉDIRE LA RÉCOLTE", use_container_width=True)

# ─── INTERFACE PRINCIPALE (SIMPLE) ──────────────────────────────────
# On propose uniquement le formulaire, le choix du modèle, le résultat et les métriques.
if predict_btn:  # Afficher l'interface uniquement après clic sur PRÉDIRE
    results = predict_all(province, culture, saison_code, altitude, pluv, temp,
                          superficie, int(engrais), int(irrigation))
    # Encoded features for debugging / verification
    X_enc = encode_input(province, culture, saison_code, altitude, pluv, temp,
                         superficie, int(engrais), int(irrigation))
    
    chosen_key, chosen_model = MODEL_MAP[model_choice]
    chosen_res = results[model_choice]
    pred  = chosen_res["pred"]
    proba = chosen_res["proba"]
    
    # Résultat principal
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.markdown('<div class="section-header">Résultat de la Prédiction</div>', unsafe_allow_html=True)
        
        if pred == 1:
            st.markdown(f"""
            <div class="pred-good">
                <p class="pred-title" style="color:#00c853">✅ BONNE RÉCOLTE</p>
                <p class="pred-subtitle">{province} — {culture}</p>
                <p class="pred-proba" style="color:#00c853">{proba*100:.1f}%</p>
                <p class="pred-subtitle">Probabilité de bonne récolte</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-bad">
                <p class="pred-title" style="color:#f44336">⚠️ MAUVAISE RÉCOLTE</p>
                <p class="pred-subtitle">{province} — {culture}</p>
                <p class="pred-proba" style="color:#f44336">{proba*100:.1f}%</p>
                <p class="pred-subtitle">Probabilité de bonne récolte</p>
            </div>
            """, unsafe_allow_html=True)
        
    with col_side:
        # Gauge simple
        fig_gauge = make_gauge(proba, f"Probabilité ({chosen_key.replace('_',' ').title()})")
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_main")
        
        # Métriques du modèle sélectionné (seulement Accuracy, F1, AUC)
        m = METRICS[chosen_key]
        st.markdown('<div class="section-header" style="font-size:0.9rem">Métriques du Modèle</div>', unsafe_allow_html=True)
        cols_m = st.columns(3)
        with cols_m[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["accuracy"]:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
        with cols_m[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["f1"]:.1%}</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
        with cols_m[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["auc"]:.3f}</div><div class="metric-label">AUC</div></div>', unsafe_allow_html=True)

    # debug removed per user request




# Footer removed per user request