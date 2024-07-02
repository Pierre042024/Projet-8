# Importation des bibliothèques nécessaires
import streamlit as st
import requests
import pandas as pd

# Titre du Dashboard
st.title("Dashboard de Scoring Client")

# Sélection du client via ID
client_id = st.text_input("Entrer l'ID du client")

# Bouton pour récupérer les informations du client
if st.button("Afficher les informations du client"):
    # Vérifie si l'ID du client n'est pas vide
    if client_id:
        # Appel à l'API pour récupérer les données du client
        response = requests.get(f"https://projet-7-service.onrender.com/predict", params={"client_id": client_id})
        
        # Vérifie si la requête est réussie
        if response.status_code == 200:
            client_data = response.json()
            
            # Affichage des données du client
            st.write("Prédiction :", client_data.get("prediction", "Non disponible"))
            st.write("Probabilité :", client_data.get("probability", "Non disponible"))
            
            # Interprétation basée sur le score
            if client_data.get("prediction") == 0:
                interpretation = "Client à risque faible"
            else:
                interpretation = "Client à risque élevé"
            st.write("Interprétation :", interpretation)
        else:
            st.error("Erreur lors de la récupération des données du client. Vérifiez l'ID et réessayez.")
    else:
        st.warning("Veuillez entrer un ID de client.")