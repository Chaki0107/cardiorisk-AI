"""CardioRisk AI Explorer - application Streamlit d'aide à la décision.

Lancer avec :  streamlit run app.py
Prérequis   :  python train.py  (génère models/model_heart.pkl et feature_importances.json)
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model_heart.pkl"
IMPORTANCES_PATH = ROOT / "models" / "feature_importances.json"

# Même liste et même ordre que dans train.py
FEATURES = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI",
            "heartRate", "glucose", "currentSmoker", "BPMeds",
            "prevalentHyp", "diabetes", "Gender_enc",
            "prevalentStroke_enc", "education_enc"]

LABELS = {"age": "Âge", "cigsPerDay": "Cigarettes / jour", "totChol": "Cholestérol total",
          "sysBP": "Pression systolique", "diaBP": "Pression diastolique", "BMI": "IMC",
          "heartRate": "Pouls", "glucose": "Glycémie", "currentSmoker": "Fumeur actuel",
          "BPMeds": "Traitement anti-HTA", "prevalentHyp": "Hypertension",
          "diabetes": "Diabète", "Gender_enc": "Genre", "prevalentStroke_enc": "AVC préalable",
          "education_enc": "Éducation"}

EDUCATION = {"Non scolarisé": 0, "École primaire": 1, "Diplômé": 2, "Post-diplômé": 3}

# Seuils indicatifs sur la probabilité calibrée (en %)
SEUIL_MODERE = 15
SEUIL_ELEVE = 35

st.set_page_config(page_title="CardioRisk AI Explorer", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)


# ── Chargement du modèle ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(IMPORTANCES_PATH, encoding="utf-8") as f:
        importances = json.load(f)
    return model, importances


if not (MODEL_PATH.exists() and IMPORTANCES_PATH.exists()):
    st.error("⚠️ Modèle introuvable. Lance d'abord `python train.py` pour générer "
             "`models/model_heart.pkl` et `models/feature_importances.json`.")
    st.stop()

model, importances = load_assets()

# ── Sidebar : saisie des données ──────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Dossier patient")

    with st.expander("👤 Informations personnelles", expanded=True):
        age = st.slider("Âge", 30, 85, 50)
        gender = st.selectbox("Genre", ["Femme", "Homme"])
        education = st.selectbox("Éducation", list(EDUCATION))

    with st.expander("🚬 Mode de vie"):
        smoker = st.radio("Fumeur actuel ?", ["Non", "Oui"], horizontal=True)
        cigs = st.number_input("Cigarettes / jour", 0, 100, 10, disabled=(smoker == "Non"))
        if smoker == "Non":
            cigs = 0

    with st.expander("🩺 Paramètres médicaux"):
        sys_bp = st.slider("Pression systolique (mmHg)", 80, 250, 120)
        dia_bp = st.slider("Pression diastolique (mmHg)", 40, 140, 80)
        tot_chol = st.number_input("Cholestérol total (mg/dL)", 100, 500, 200)
        bmi = st.number_input("IMC (kg/m²)", 10.0, 50.0, 25.0)
        glucose = st.number_input("Glycémie (mg/dL)", 50, 400, 90)
        heart_rate = st.slider("Pouls (bpm)", 40, 160, 75)

    with st.expander("💊 Antécédents"):
        bp_meds = st.checkbox("Prend des médicaments contre l'HTA")
        stroke = st.checkbox("A déjà fait un AVC")
        hyp = st.checkbox("Souffre d'hypertension")
        diabetes = st.checkbox("Est diabétique")

# ── Page principale ───────────────────────────────────────────────────────────
st.title("🫀 CardioRisk AI Explorer")
st.markdown("Aide à la décision : estimation du risque cardiovasculaire à partir de "
            "données cliniques (modèle entraîné sur l'étude de Framingham).")

row = {"age": age, "cigsPerDay": cigs, "totChol": tot_chol, "sysBP": sys_bp,
       "diaBP": dia_bp, "BMI": bmi, "heartRate": heart_rate, "glucose": glucose,
       "currentSmoker": int(smoker == "Oui"), "BPMeds": int(bp_meds),
       "prevalentHyp": int(hyp), "diabetes": int(diabetes),
       "Gender_enc": int(gender == "Homme"), "prevalentStroke_enc": int(stroke),
       "education_enc": EDUCATION[education]}
patient = pd.DataFrame([row], columns=FEATURES)

if st.button("🚀 ANALYSER LE RISQUE", width="stretch", type="primary"):
    # Le pipeline du modèle applique lui-même la standardisation
    proba = float(model.predict_proba(patient)[0, 1]) * 100

    st.markdown("---")
    col_gauge, col_diag = st.columns(2)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=proba, number={"suffix": " %"},
            title={"text": "Probabilité estimée (%)", "font": {"size": 22}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "black"},
                   "steps": [{"range": [0, SEUIL_MODERE], "color": "#2ecc71"},
                             {"range": [SEUIL_MODERE, SEUIL_ELEVE], "color": "#f1c40f"},
                             {"range": [SEUIL_ELEVE, 100], "color": "#e74c3c"}]}))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

    with col_diag:
        st.write("### Résultat")
        if proba < SEUIL_MODERE:
            st.success("#### ✅ RISQUE FAIBLE")
            st.write("Le profil du patient ne présente pas de signal d'alerte majeur.")
            conseil = "Maintenir une activité physique régulière et un suivi de routine."
        elif proba < SEUIL_ELEVE:
            st.warning("#### ⚠️ RISQUE MODÉRÉ")
            st.write("Certains facteurs pèsent sur le score global : une surveillance est conseillée.")
            conseil = "Réduire le sel et le tabac, et discuter d'un suivi avec un médecin."
        else:
            st.error("#### 🚨 RISQUE ÉLEVÉ")
            st.write("Le modèle estime une probabilité élevée d'événement cardiovasculaire.")
            conseil = "Une consultation médicale spécialisée est recommandée."
        st.info(f"💡 **Conseil général :** {conseil}")

    st.markdown("### 🔍 Variables les plus influentes du modèle")
    st.caption("Importance globale (mesurée sur l'ensemble des patients), "
               "et non spécifique à ce patient.")
    feat_df = (pd.DataFrame({"Variable": [LABELS[k] for k in importances],
                             "Importance": list(importances.values())})
               .sort_values("Importance").tail(8))
    fig_imp = px.bar(feat_df, x="Importance", y="Variable", orientation="h",
                     color="Importance", color_continuous_scale="Reds")
    st.plotly_chart(fig_imp, width="stretch")
else:
    st.info("Renseigne le dossier du patient à gauche, puis clique sur « Analyser le risque ».")

st.markdown("---")
st.caption("⚠️ Prototype pédagogique : cet outil ne remplace pas le diagnostic d'un "
           "professionnel de santé. Les données saisies ne sont pas enregistrées.")
