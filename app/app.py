import streamlit as st
import pandas as pd
import requests

API_URL = "https://credit-scoring-api-tqja.onrender.com/predict"
API_CLIENTS_URL = "https://credit-scoring-api-tqja.onrender.com/clients"


st.title("Credit Scoring")

try:
    response = requests.get(API_CLIENTS_URL)
    response.raise_for_status()
    clients_df = pd.DataFrame.from_dict(response.json(), orient="index")
except Exception as e:
    st.error(f"Erreur lors de la récupération des clients : {e}")
    st.stop()


st.subheader("Sélection du client")
client_id = st.selectbox("Choisir un client :", clients_df.index.tolist())
client_data = clients_df.loc[client_id]


predict_button = st.button("Prédire")


st.subheader("Données du client")
inputs = {}
for col in clients_df.columns:
    val = client_data[col]
    try:
        val = float(val)
    except ValueError:
        val = 0.0
    inputs[col] = st.number_input(col, value=val)


if predict_button:
    try:
        res = requests.post(API_URL, json=inputs).json()
        if "error" in res:
            st.error(f"Erreur API : {res['error']}")
        else:
            proba = res["proba"]
            classe = res["classe"]
            st.write(f"Probabilité de défaut : {proba:.2f}")
            if classe == 1:
                st.error("Crédit refusé")
            else:
                st.success("Crédit accordé")
    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")






