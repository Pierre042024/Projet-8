# Importation des bibliothèques nécessaires
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Titre du Dashboard
st.title("Dashboard de Scoring Client")

# Sélection du client via ID
client_id = st.text_input("Entrer l'ID du client")

# Charger les données des clients pour la visualisation des distributions
clients_df = pd.read_csv('sampled_dataset.csv')

# Sélectionner les features disponibles pour la visualisation
available_features = clients_df.drop(columns=['SK_ID_CURR', 'TARGET']).columns.tolist()

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
        
# Bouton pour récupérer les features les plus importants pour le client
if st.button("Afficher les features les plus importants"):
    # Vérifie si l'ID du client n'est pas vide
    if client_id:
        # Appel à l'API pour récupérer les features les plus importants
        response = requests.get(f"https://projet-7-service.onrender.com/top_features", params={"client_id": client_id})
        
        # Vérifie si la requête est réussie
        if response.status_code == 200:
            top_features = response.json()
            
            # Affichage des features les plus importants
            st.write("Top 10 features les plus importants pour ce client :")
            
            # Préparer les données pour le graphique
            features = [feature['feature'] for feature in top_features]
            scores = [feature['score'] for feature in top_features]
            data = pd.DataFrame({'Feature': features, 'Score': scores})
            
            # Créer le graphique avec Plotly
            fig = px.bar(data, x='Score', y='Feature', orientation='h', title="Top 10 features les plus importants")
            st.plotly_chart(fig)
        else:
            st.error("Erreur lors de la récupération des features. Vérifiez l'ID et réessayez.")
    else:
        st.warning("Veuillez entrer un ID de client.")

# Sélectionner plusieurs features à visualiser
selected_features = st.multiselect("Sélectionnez les features à visualiser", available_features)        

# Fonction pour afficher les distributions des features
def plot_feature_distribution(feature, client_value):
    fig = px.histogram(clients_df, x=feature, color='TARGET', barmode='overlay', title=f"Distribution de {feature}")
    fig.add_vline(x=client_value, line_width=3, line_dash="dash", line_color="red", annotation_text="Client", annotation_position="top right")
    return fig

# Bouton pour afficher les distributions des features sélectionnées
if st.button("Afficher les distributions des features"):
    # Vérifie si l'ID du client n'est pas vide
    if client_id:
        # Extraire les données du client
        client_data = clients_df[clients_df['SK_ID_CURR'] == int(client_id)]
        
        if not client_data.empty:
            for feature in selected_features:
                client_value = client_data[feature].values[0]
                
                # Afficher les graphiques de distribution
                st.write(f"Distribution de la feature {feature}")
                fig = px.histogram(clients_df, x=feature, color='TARGET', marginal="box",
                                   title=f"Distribution de {feature} par classe")
                fig.add_vline(x=client_value, line_dash="dash", line_color="red", annotation_text="Client", annotation_position="top right")
                st.plotly_chart(fig)
        else:
            st.error("Client non trouvé. Veuillez vérifier l'ID du client.")
    else:
        st.warning("Veuillez entrer un ID de client.")
        
# Fonction pour afficher le graphique d'analyse bi-variée
def plot_bivariate_analysis(feature1, feature2, client_id):
    fig = px.scatter(clients_df, x=feature1, y=feature2, color='TARGET', title=f"Analyse bi-variée : {feature1} vs {feature2}", labels={feature1: feature1, feature2: feature2}, color_continuous_scale=px.colors.sequential.Viridis)
    
    # Ajouter le point du client
    client_data = clients_df[clients_df['SK_ID_CURR'] == int(client_id)]
    if not client_data.empty:
        client_value1 = client_data[feature1].values[0]
        client_value2 = client_data[feature2].values[0]
        fig.add_trace(go.Scatter(x=[client_value1], y=[client_value2], mode='markers', marker=dict(color='red', size=12), name='Client'))
    
    return fig

# Bouton pour afficher l'analyse bi-variée
if st.button("Afficher l'analyse bi-variée"):
    # Vérifie si l'ID du client n'est pas vide
    if client_id:
        st.write(f"Analyse bi-variée entre {selected_feature1} et {selected_feature2}")
        fig = plot_bivariate_analysis(selected_feature1, selected_feature2, client_id)
        st.plotly_chart(fig)
    else:
        st.warning("Veuillez entrer un ID de client.")
        
# Bouton pour récupérer l'importance globale des features
if st.button("Afficher l'importance globale des features"):
    # Appel à l'API pour récupérer l'importance globale des features
    response = requests.get("https://projet-8-3xet.onrender.com/global_feature_importance")
    
    # Vérifie si la requête est réussie
    if response.status_code == 200:
        try:
            global_importance = response.json()
            
            # Préparer les données pour le graphique
            features = [feature['feature'] for feature in global_importance]
            importances = [feature['importance'] for feature in global_importance]
            data = pd.DataFrame({'Feature': features, 'Importance': importances})
            
            # Créer le graphique avec Plotly
            fig = px.bar(data, x='Importance', y='Feature', orientation='h', title="Importance globale des features")
            st.plotly_chart(fig)
        
        except json.JSONDecodeError as e:
            st.error(f"Erreur JSON lors de la récupération de l'importance globale des features : {e}")
        
        except Exception as e:
            st.error(f"Erreur inattendue lors de la récupération de l'importance globale des features : {e}")
    
    else:
        st.error("Erreur lors de la récupération de l'importance globale des features.")