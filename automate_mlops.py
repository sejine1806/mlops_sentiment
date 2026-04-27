import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "Mamba_SSM_Production"

# 1. Récupérer la version actuelle en 'staging' (Le Champion)
try:
    champion_version = client.get_model_version_by_alias(model_name, "staging")
    champion_run_id = champion_version.run_id
    champion_loss = client.get_run(champion_run_id).data.metrics.get("mse_loss", float('inf'))
    print(f"🏆 Champion actuel: Version {champion_version.version} (Loss: {champion_loss})")
except:
    champion_loss = float('inf')
    print("🏆 Aucun champion en staging pour le moment.")

# 2. Récupérer le dernier run de l'expérience 'Default' (Le Challenger)
# On cherche le run le plus récent
experiment = client.get_experiment_by_name("Default") # Ou le nom de ton exp
last_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=1,
    order_by=["attributes.start_time DESC"]
)[0]

challenger_loss = last_run.data.metrics.get("mse_loss", float('inf'))
challenger_run_id = last_run.info.run_id
print(f"🚀 Challenger: Run {challenger_run_id} (Loss: {challenger_loss})")

# 3. Comparaison et Promotion
if challenger_loss < champion_loss:
    print("✅ Le Challenger est meilleur ! Enregistrement et promotion...")
    
    # Enregistrer le nouveau modèle
    model_uri = f"runs:/{challenger_run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    
    # Lui attribuer l'alias 'staging'
    client.set_registered_model_alias(model_name, "staging", mv.version)
    print(f"✨ Version {mv.version} est maintenant en STAGING.")
else:
    print("❌ Le Challenger n'est pas meilleur. On garde le Champion.")
