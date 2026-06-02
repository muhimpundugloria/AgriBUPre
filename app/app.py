

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
def encode_input(province, culture, saison, altitude, pluv, temp, superficie,
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
    if saison == "B":
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
        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
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
    saison    = st.selectbox("📅 Saison",    ["A (mars–juin)", "B (sept–déc)"])
    saison_code = "A" if saison.startswith("A") else "B"
    
    st.markdown("---")
    altitude  = st.slider("⛰️ Altitude (m)",      400, 2600, 1720, step=10)
    pluv      = st.slider("🌧️ Pluviométrie (mm)", 150, 1600, 900,  step=10)
    temp      = st.slider("🌡️ Température (°C)",  14.0, 30.0, 18.2, step=0.1)
    superficie= st.slider("📐 Superficie (ha)",   5.0, 800.0, 50.0, step=5.0)
    
    st.markdown("---")
    engrais   = st.toggle("🧪 Utilisation d'engrais",  value=True)
    irrigation= st.toggle("💧 Accès à l'irrigation",   value=False)
    
    st.markdown("---")
    model_choice = st.selectbox(
        "🤖 Modèle",
        list(MODEL_MAP.keys()),
        index=1
    )
    
    predict_btn = st.button("🔍  PRÉDIRE LA RÉCOLTE", use_container_width=True)

# ─── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  Prédiction",
    "📊  Analyse des Modèles",
    "📈  Comparaison ROC",
    "🌍  Scénarios Burundi",
    "ℹ️  À propos"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    if predict_btn or True:  # Always show the interface
        results = predict_all(province, culture, saison_code, altitude, pluv, temp,
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
                    <p class="pred-subtitle">{province} — {culture} — Saison {saison_code}</p>
                    <p class="pred-proba" style="color:#00c853">{proba*100:.1f}%</p>
                    <p class="pred-subtitle">Probabilité de bonne récolte</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-bad">
                    <p class="pred-title" style="color:#f44336">⚠️ MAUVAISE RÉCOLTE</p>
                    <p class="pred-subtitle">{province} — {culture} — Saison {saison_code}</p>
                    <p class="pred-proba" style="color:#f44336">{proba*100:.1f}%</p>
                    <p class="pred-subtitle">Probabilité de bonne récolte</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Conseils agronomiques
            st.markdown("<br>", unsafe_allow_html=True)
            tips = []
            if pluv < 500:
                tips.append("💧 Pluviométrie très faible — envisager l'irrigation d'appoint ou cultures résistantes à la sécheresse")
            if pluv > 1300:
                tips.append("🌊 Pluviométrie excessive — risque de lessivage et maladies fongiques")
            if not engrais:
                tips.append("🧪 L'utilisation d'engrais augmente significativement les rendements (+35% en moyenne)")
            if not irrigation and pluv < 700:
                tips.append("💦 Accès à l'irrigation recommandé avec cette pluviométrie")
            if temp > 27:
                tips.append("🌡️ Températures élevées — risque de stress thermique pour cette culture")
            if superficie > 300:
                tips.append("📐 Grande superficie — mécanisation agricole recommandée")
            
            if tips:
                st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1rem">💡 Recommandations Agronomiques</div>', unsafe_allow_html=True)
                for t in tips:
                    st.markdown(f'<div class="info-box">{t}</div>', unsafe_allow_html=True)
        
        with col_side:
            # Gauge
            fig_gauge = make_gauge(proba, f"Probabilité ({chosen_key.replace('_',' ').title()})")
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_main")
            
            # Métriques du modèle sélectionné
            m = METRICS[chosen_key]
            st.markdown('<div class="section-header" style="font-size:0.9rem">Métriques du Modèle</div>', unsafe_allow_html=True)
            
            cols_m = st.columns(2)
            with cols_m[0]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{m["accuracy"]:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-value">{m["f1"]:.1%}</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
            with cols_m[1]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{m["auc"]:.3f}</div><div class="metric-label">AUC-ROC</div></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card"><div class="metric-value">{m["precision"]:.1%}</div><div class="metric-label">Précision</div></div>', unsafe_allow_html=True)
        
        # Comparaison des 3 modèles
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Comparaison des 3 Modèles sur cette Parcelle</div>', unsafe_allow_html=True)
        
        cols3 = st.columns(3)
        icons = ["🌳", "🌲", "📈"]
        for i, (lbl, res) in enumerate(results.items()):
            with cols3[i]:
                fig_g = make_gauge(res["proba"], lbl.split(" ", 1)[1])
                st.plotly_chart(fig_g, use_container_width=True, key=f"gauge_{i}")
                verdict = "✅ Bonne" if res["pred"] == 1 else "⚠️ Mauvaise"
                color   = "#00c853" if res["pred"] == 1 else "#f44336"
                st.markdown(f'<p style="text-align:center;font-weight:900;color:{color};font-size:1.1rem">{verdict}</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE DES MODÈLES
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Performance Globale des Modèles</div>', unsafe_allow_html=True)
    
    # Métriques comparatives
    metric_names = ["Accuracy", "F1-Score", "AUC-ROC", "Précision", "Rappel"]
    model_display = {
        "decision_tree": "Arbre de Décision",
        "random_forest": "Forêt Aléatoire",
        "logistic_regression": "Régression Logistique"
    }
    
    df_metrics = pd.DataFrame([
        {
            "Modèle": model_display[k],
            "Accuracy": f"{v['accuracy']:.1%}",
            "F1-Score": f"{v['f1']:.1%}",
            "AUC-ROC": f"{v['auc']:.3f}",
            "Précision": f"{v['precision']:.1%}",
            "Rappel": f"{v['recall']:.1%}",
        }
        for k, v in METRICS.items()
    ])
    st.dataframe(df_metrics.set_index("Modèle"), use_container_width=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Feature importance
        st.markdown('<div class="section-header" style="font-size:1rem">Importance des Variables — Forêt Aléatoire</div>', unsafe_allow_html=True)
        rf_fi = meta["rf_feature_importances"]
        top20 = sorted(rf_fi.items(), key=lambda x: x[1], reverse=True)[:15]
        feats, vals = zip(*top20)
        
        # Clean feature names for display
        def clean_name(n):
            n = n.replace("province_", "Province: ").replace("culture_", "Culture: ")
            n = n.replace("saison_", "Saison ").replace("_", " ").title()
            return n
        
        feats_clean = [clean_name(f) for f in feats]
        
        fig_fi = go.Figure(go.Bar(
            x=list(vals), y=feats_clean,
            orientation='h',
            marker=dict(
                color=list(vals),
                colorscale=[[0, "#1e4d6b"], [0.5, "#00b4d8"], [1, "#64ffda"]],
                showscale=False
            )
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=450,
            title=dict(text="Top 15 Features", font=dict(color="#64ffda", size=13)))
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True, key="fi_rf")
    
    with col_b:
        # Overfitting curve
        st.markdown('<div class="section-header" style="font-size:1rem">Overfitting — Arbre de Décision</div>', unsafe_allow_html=True)
        ov = meta["overfitting"]
        
        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(
            x=ov["depths"], y=ov["train_scores"], name="Train",
            line=dict(color="#64ffda", width=2.5), mode="lines+markers",
            marker=dict(size=6)
        ))
        fig_ov.add_trace(go.Scatter(
            x=ov["depths"], y=ov["test_scores"], name="Test",
            line=dict(color="#f48c06", width=2.5), mode="lines+markers",
            marker=dict(size=6)
        ))
        best = np.argmax(ov["test_scores"]) + 1
        fig_ov.add_vline(x=best, line_dash="dash", line_color="#ff6b6b",
                         annotation_text=f"Best depth={best}", annotation_font_color="#ff6b6b")
        fig_ov.update_layout(**PLOTLY_LAYOUT, height=300,
            xaxis_title="Profondeur max", yaxis_title="Accuracy",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(100,255,218,0.2)"))
        st.plotly_chart(fig_ov, use_container_width=True, key="overfitting")
        
        # Matrice de confusion RF
        st.markdown('<div class="section-header" style="font-size:1rem">Matrice de Confusion — Forêt Aléatoire</div>', unsafe_allow_html=True)
        cm = METRICS["random_forest"]["confusion_matrix"]
        labels = ["Mauvaise Récolte", "Bonne Récolte"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#0a1628"], [1, "#64ffda"]],
            showscale=False,
            text=cm, texttemplate="%{text}",
            textfont={"size": 22, "color": "white"}
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=260,
            xaxis_title="Prédit", yaxis_title="Réel")
        st.plotly_chart(fig_cm, use_container_width=True, key="conf_mat")

# ══════════════════════════════════════════════════════════════
# TAB 3 — COURBES ROC
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Courbes ROC — Comparaison des 3 Modèles</div>', unsafe_allow_html=True)
    
    colors = {"decision_tree": "#f48c06", "random_forest": "#64ffda", "logistic_regression": "#e040fb"}
    names  = {"decision_tree": "Arbre de Décision", "random_forest": "Forêt Aléatoire", "logistic_regression": "Régression Logistique"}
    
    fig_roc = go.Figure()
    # Ligne diagonale
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="#495670", width=1.5),
        name="Aléatoire (AUC=0.5)", showlegend=True))
    
    for key, m in METRICS.items():
        fig_roc.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"],
            mode="lines", name=f"{names[key]} (AUC={m['auc']:.3f})",
            line=dict(color=colors[key], width=2.5)
        ))
    
    fig_roc.update_layout(**PLOTLY_LAYOUT, height=500,
        xaxis_title="Taux de Faux Positifs (FPR)",
        yaxis_title="Taux de Vrais Positifs (TPR)",
        title=dict(text="Courbes ROC — Prédiction de Bonne Récolte", font=dict(color="#64ffda", size=15)),
        legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="rgba(100,255,218,0.2)", x=0.55, y=0.1))
    fig_roc.update_xaxes(range=[0, 1])
    fig_roc.update_yaxes(range=[0, 1])
    
    st.plotly_chart(fig_roc, use_container_width=True, key="roc")
    
    # Explication
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    for col, (key, m) in zip([col_exp1, col_exp2, col_exp3], METRICS.items()):
        with col:
            auc_v = m["auc"]
            quality = "🏆 Excellent" if auc_v > 0.9 else ("✅ Bon" if auc_v > 0.8 else "⚠️ Acceptable")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{colors[key]}">{auc_v:.3f}</div>
                <div class="metric-label">AUC — {names[key]}</div>
                <div style="color:{colors[key]};font-size:0.9rem;margin-top:0.5rem;font-weight:700">{quality}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="info-box" style="margin-top:1.5rem">📚 <strong>Interprétation AUC :</strong> L\'AUC (Area Under Curve) mesure la capacité du modèle à discriminer entre bonnes et mauvaises récoltes, indépendamment du seuil de décision. AUC=1 = parfait, AUC=0.5 = aléatoire. La Forêt Aléatoire (AUC≈0.93) est nettement supérieure aux autres modèles.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — SCÉNARIOS BURUNDI
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Scénarios Agricoles Réels au Burundi (Exercice 6)</div>', unsafe_allow_html=True)
    
    SCENARIOS = [
        {"label": "🌽 Kayanza – Maïs",       "province": "Kayanza",  "culture": "Maïs",        "altitude": 1980, "pluv": 920,  "temp": 17.8, "engrais": True,  "irrigation": False, "superficie": 45},
        {"label": "🥔 Bubanza – Manioc",      "province": "Bubanza",  "culture": "Manioc",      "altitude": 790,  "pluv": 550,  "temp": 25.4, "engrais": False, "irrigation": False, "superficie": 30},
        {"label": "🫘 Gitega – Haricot",       "province": "Gitega",   "culture": "Haricot",     "altitude": 1720, "pluv": 430,  "temp": 18.2, "engrais": False, "irrigation": False, "superficie": 20},
        {"label": "🍠 Cibitoke – Patate douce","province": "Cibitoke", "culture": "Patate douce","altitude": 810,  "pluv": 810,  "temp": 24.1, "engrais": True,  "irrigation": True,  "superficie": 55},
    ]
    
    all_results = []
    for s in SCENARIOS:
        r = predict_all(s["province"], s["culture"], "A", s["altitude"],
                        s["pluv"], s["temp"], s["superficie"],
                        int(s["engrais"]), int(s["irrigation"]))
        all_results.append(r)
    
    # Tableau récap
    table_rows = []
    for s, r in zip(SCENARIOS, all_results):
        row = {"Scénario": s["label"]}
        for lbl, res in r.items():
            short = lbl.split(" ", 1)[1]
            verdict = "✅ Bonne" if res["pred"] == 1 else "⚠️ Mauvaise"
            row[short] = f"{verdict} ({res['proba']*100:.0f}%)"
        table_rows.append(row)
    
    df_tab = pd.DataFrame(table_rows).set_index("Scénario")
    st.dataframe(df_tab, use_container_width=True)
    
    # Détail par scénario
    st.markdown("<br>", unsafe_allow_html=True)
    for s, r in zip(SCENARIOS, all_results):
        with st.expander(f"📋 Détail — {s['label']}"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("⛰️ Altitude", f"{s['altitude']} m")
            with c2: st.metric("🌧️ Pluviométrie", f"{s['pluv']} mm")
            with c3: st.metric("🌡️ Température", f"{s['temp']} °C")
            with c4: st.metric("🧪 Engrais", "Oui" if s["engrais"] else "Non")
            
            cols_g = st.columns(3)
            for i, (lbl, res) in enumerate(r.items()):
                with cols_g[i]:
                    fig_g = make_gauge(res["proba"], lbl.split(" ", 1)[1])
                    st.plotly_chart(fig_g, use_container_width=True, key=f"scen_{s['province']}_{i}")
            
            # Alerte spéciale scénario 3
            if s["province"] == "Gitega" and s["culture"] == "Haricot":
                st.markdown("""
                <div class="warning-box">
                ⚠️ <strong>Attention :</strong> La pluviométrie de 430mm est très insuffisante pour le Haricot (optimum 700–900mm). 
                Les modèles prédisent une mauvaise récolte avec forte confiance. 
                Recommandations : mise en place de l'irrigation, ou report de la saison de semis, ou substitution par du Sorgho (plus résistant à la sécheresse).
                </div>
                """, unsafe_allow_html=True)
    
    # Analyse désaccords
    st.markdown('<div class="section-header" style="margin-top:2rem">Analyse des Désaccords entre Modèles</div>', unsafe_allow_html=True)
    unanimes = []
    discords = []
    for s, r in zip(SCENARIOS, all_results):
        preds = [res["pred"] for res in r.values()]
        if len(set(preds)) == 1:
            unanimes.append(s["label"])
        else:
            discords.append(s["label"])
    
    if unanimes:
        st.markdown(f'<div class="info-box">✅ <strong>Unanimité :</strong> {", ".join(unanimes)}</div>', unsafe_allow_html=True)
    if discords:
        st.markdown(f'<div class="warning-box">⚡ <strong>Désaccords :</strong> {", ".join(discords)} — En cas de désaccord, privilégier la Forêt Aléatoire (meilleure AUC) ou adopter une approche de vote majoritaire.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — À PROPOS
# ══════════════════════════════════════════════════════════════
with tab5:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-header">📚 Contexte du Projet</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Ce projet a été développé dans le cadre du cours <strong>IA Appliquée à l'Agriculture</strong> 
        — BAC 4 Génie Logiciel, Université Polytechnique de Gitega.
        <br><br>
        L'agriculture représente plus de 40% du PIB du Burundi et constitue la principale source 
        de revenus pour plus de 90% de la population rurale. Ce système d'aide à la décision 
        vise à anticiper les bonnes et mauvaises récoltes.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🗃️ Dataset</div>', unsafe_allow_html=True)
        info = meta.get("dataset_info", {})
        
        cols_ds = st.columns(2)
        with cols_ds[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info.get("total_rows", 1579)}</div><div class="metric-label">Observations</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info.get("provinces", 15)}</div><div class="metric-label">Provinces</div></div>', unsafe_allow_html=True)
        with cols_ds[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Cultures</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">2015–23</div><div class="metric-label">Période</div></div>', unsafe_allow_html=True)
    
    with col_b:
        st.markdown('<div class="section-header">🤖 Modèles Implémentés</div>', unsafe_allow_html=True)
        for key, (disp, acc_key) in [
            ("decision_tree", ("Arbre de Décision", "decision_tree")),
            ("random_forest", ("Forêt Aléatoire", "random_forest")),
            ("logistic_regression", ("Régression Logistique", "logistic_regression")),
        ]:
            m = METRICS[key]
            st.markdown(f"""
            <div class="scenario-card">
                <strong style="color:#64ffda">{disp}</strong><br>
                <small>Accuracy : <strong>{m['accuracy']:.1%}</strong> &nbsp;|&nbsp; 
                AUC : <strong>{m['auc']:.3f}</strong> &nbsp;|&nbsp; 
                F1 : <strong>{m['f1']:.1%}</strong></small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header" style="margin-top:1.5rem">⚠️ Limites du Modèle</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-box">
        • Dataset synthétique — nécessite validation sur données terrain réelles<br>
        • Ne prend pas en compte : maladies, ravageurs, marchés, chocs climatiques extrêmes<br>
        • Classe déséquilibrée (~75% mauvaises récoltes) — biais possible<br>
        • Les prédictions sont des estimations probabilistes, pas des certitudes
        </div>
        """, unsafe_allow_html=True)
    
    # Recommandation finale
    st.markdown('<div class="section-header">🎯 Recommandation de Déploiement (Q30)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Modèle recommandé : Forêt Aléatoire</strong><br>
    La Forêt Aléatoire offre le meilleur équilibre performance/robustesse (AUC ≈ 0.93, Accuracy ≈ 88%). 
    Contrairement à l'arbre de décision seul, elle résiste au surapprentissage grâce au bagging. 
    Contrairement à la régression logistique, elle capture les relations non-linéaires dans les données agricoles.<br><br>
    <strong>Données supplémentaires souhaitables :</strong> type de sol, pH, données satellites NDVI, 
    prix des intrants, accès aux marchés, données météo quotidiennes, historique des maladies.
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AgriPredict Burundi • Université Polytechnique de Gitega • BAC 4 Génie Logiciel<br>
    Modèles : Arbre de Décision | Forêt Aléatoire | Régression Logistique • Scikit-learn + Streamlit
</div>
""", unsafe_allow_html=True)