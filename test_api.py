import requests
import json
import numpy as np


# L'adresse locale de ton serveur MLflow
url = "http://127.0.0.1:5001/invocations"

# On prépare une donnée qui correspond à ton d_model (512)
# Format attendu par MLflow : {"inputs": [ [ [valeurs] ] ]}
dummy_data = np.random.randn(1, 10, 512).astype(np.float32).tolist()
data = {"inputs": dummy_data}

print("📡 Envoi de la requête au modèle Mamba...")

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print("✅ Réponse reçue !")
        print(f"Sortie du modèle : {response.json()}")
    else:
        print(f"❌ Erreur {response.status_code}: {response.text}")
except Exception as e:
    print(f"💥 Connexion impossible : {e}")
