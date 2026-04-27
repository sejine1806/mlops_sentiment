import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import argparse
from mlflow.models import infer_signature

d_model = 128
d_state = 16

# 1. Lecture des paramètres envoyés par MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.005)
args = parser.parse_args()

# On ne fait plus de "set_experiment" ici, on laisse MLflow s'en charger
# On ne fait plus de "start_run", on utilise le run déjà actif

# --- Architecture ---
class SimpleSSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        x = x.to(self.out_proj.weight.dtype)
        return self.out_proj(x)

# --- Entraînement ---
model = SimpleSSM(args.d_model, 32)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

# Log des paramètres directement dans le run actif
mlflow.log_params({"d_model": args.d_model, "lr": args.lr})

x = torch.randn(16, 10, args.d_model)
y = torch.randn(16, 10, args.d_model)

print(f"🚀 Training Mamba (d_model={args.d_model}, lr={args.lr})")

for epoch in range(5):
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    mlflow.log_metric("mse_loss", loss.item(), step=epoch)

# Sauvegarde
mlflow.pytorch.log_model(model, "model")
print("✅ Terminé !")


# --- À la fin de ton script, après la boucle for epoch ---

# 1. Préparation du dummy_input DYNAMIQUE
# On utilise args.d_model (128 par défaut) au lieu de 512 pour que ça corresponde au modèle
dummy_input = torch.randn(1, 10, args.d_model).float() 

# 2. Test de passage (Inference)
with torch.no_grad():
    prediction = model(dummy_input)

# 3. Création de la signature
signature = infer_signature(dummy_input.numpy(), prediction.detach().numpy())

# 4. SAUVEGARDE UNIQUE (On ne garde que celle-là)
mlflow.pytorch.log_model(
    pytorch_model=model, 
    artifact_path="model", 
    signature=signature,
    input_example=dummy_input.numpy()
)

print(f"✅ Modèle loggé avec Signature (d_model={args.d_model})")
