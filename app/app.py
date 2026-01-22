import streamlit as st
import pandas as pd
import requests

API_URL = "https://credit-scoring-api-tqja.onrender.com/predict"
API_CLIENTS_URL = "https://credit-scoring-api-tqja.onrender.com/clients"

st.title("Credit Scoring")

# Charger les clients
try:
    response = requests.get(API_CLIENTS_URL)
    response.raise_for_status()
    clients_df = pd.DataFrame.from_dict(response.json(), orient="index")
except Exception as e:
    st.error(f"Erreur lors de la récupération des clients : {e}")
    st.stop()

# Sélection du client
client_id = st.selectbox("Choisir un client :", clients_df.index.tolist())
client_data = clients_df.loc[client_id]

# Création des colonnes pour bouton et résultat
col1, col2 = st.columns([1, 1])

# Préparer le formulaire
inputs = {}
for col in clients_df.columns:
    val = client_data[col]
    try:
        val = float(val)
    except ValueError:
        val = 0.0
    inputs[col] = st.number_input(col, value=val)

# Bouton de prédiction en haut à gauche
with col1:
    predict_button = st.button("Prédire")

# Résultat en haut à droite
result_container = col2.empty()

if predict_button:
    try:
        res = requests.post(API_URL, json=inputs).json()
        if "error" in res:
            result_container.error(f"Erreur API : {res['error']}")
        else:
            classe = res["classe"]
            if classe == 1:
                result_container.error("Crédit refusé ❌")
            else:
                result_container.success("Crédit accordé ✅")
    except Exception as e:
        result_container.error(f"Erreur lors de l'appel API : {e}")










