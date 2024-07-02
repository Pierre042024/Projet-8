from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import shap
import numpy as np

# Charger le modèle enregistré
model = joblib.load('best_model.joblib')

# Charger les données des clients avec SK_ID_CURR comme index
try:
    clients_df = pd.read_csv('sampled_dataset.csv')
except ValueError as e:
    raise ValueError(f"Error reading CSV file: {e}")

# Initialiser l'application FastAPI
app = FastAPI()

# Route pour vérifier l'état de l'API
@app.get("/")
def read_root():
    return {"status": "API is running"}

# Route pour afficher les scores de précision du modèle
@app.get("/model_scores")
def model_scores():
    # Placeholder scores
    scores = {
        "auc": 0.74,
        "accuracy": 0.93,
    }
    return scores

# Route pour obtenir la liste des IDs des clients
@app.get("/clients")
def get_client_ids():
    return clients_df['SK_ID_CURR'].tolist()

# Route pour faire une prédiction basée sur l'ID du client
@app.get("/predict")
async def predict(client_id: int):
    # Vérifier si le client existe
    if client_id not in clients_df['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Extraire les données du client et supprimer la colonne 'SK_ID_CURR'
    client_data = clients_df[clients_df['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR', 'TARGET'])
    
    # Faire une prédiction
    prediction = model.predict(client_data.values.reshape(1, -1))
    if prediction[0] == 0:
        prediction_proba = model.predict_proba(client_data.values.reshape(1, -1))[0, 0]
    else:
        prediction_proba = model.predict_proba(client_data.values.reshape(1, -1))[0, 1]

    # Retourner la prédiction
    return {
        "client_id": client_id,
        "prediction": int(prediction[0]),
        "probability": float(prediction_proba)
    }

# Route pour afficher les 10 features les plus importants avec leurs scores
@app.get("/top_features")
def get_top_features(client_id: int):
    # Vérifier si le client existe
    if client_id not in clients_df['SK_ID_CURR'].values:
        raise HTTPException(status_code=404, detail="Client not found")

    # Extraire les données du client et supprimer la colonne 'SK_ID_CURR'
    client_data = clients_df[clients_df['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR', 'TARGET'])
    
    # Calculer les SHAP values pour cette instance
    #explainer = shap.Explainer(model, clients_df.drop(columns=['SK_ID_CURR'])) #erreur, lui donner uniquement le classifier du pipeline et aussi retiré la colonne 'TARGET'
    #shap_values = explainer(client_data)
    
    # Extraire le modèle LightGBM du pipeline
    lgbm_scorer = model.named_steps['classifier']

    # Créer le DataFrame à partir des données de test
    X_test_df = clients_df.drop(columns=['SK_ID_CURR', 'TARGET'])

    # Fits the explainer
    explainer = shap.TreeExplainer(lgbm_scorer)

    # Calculates the SHAP values for the instance
    shap_values = explainer.shap_values(client_data)

    # Extraire les noms des features
    feature_names = client_data.columns.tolist()

    # Extraire les SHAP values pour l'instance sélectionnée
    shap_values_instance = shap_values[1][0]

    # Calculer les scores absolus des SHAP values
    abs_shap_values = np.abs(shap_values_instance)

    # Trier les indices par importance (en utilisant les valeurs absolues)
    sorted_indices = np.argsort(abs_shap_values)[::-1]

    # Sélectionner les 10 features les plus importants
    top_indices = sorted_indices[:10]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [shap_values_instance[i] for i in top_indices]

    # Créer un dictionnaire pour le tableau JSON
    result_table = [{"feature": feature, "score": score} for feature, score in zip(top_features, top_scores)]

    return result_table

# Exécution du serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)