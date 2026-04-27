import mlflow.pytorch
import torch

model_name = "Mamba_SSM_Production"
model_version = "staging"

print(f"📡 Chargement du modèle {model_name} ({model_version})...")
model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_version}")

test_input = torch.randn(1, 10, 512)

#Prediction
model.eval()
with torch.no_grad():
    prediction = model(test_input)
    print("✅ Prédiction réussie !")
    print(f"Sortie du modèle (forme) : {prediction.shape}")
