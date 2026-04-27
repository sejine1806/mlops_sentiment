import subprocess
import sys

def run_step(command, title):
    print(f"\n--- 🛠️  {title} ---")
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"❌ Erreur critique lors de : {title}. Arrêt du pipeline.")
        sys.exit(1)

def main():
    print("🚀 DÉMARRAGE DU PIPELINE AUTOMATIQUE")

    # Étape 1 : Entraînement (Challenger)
    # On force les paramètres ici
    run_step("mlflow run . -P d_model=512 --env-manager local", "ENTRAÎNEMENT")

    # Étape 2 : Comparaison et Promotion (Champion vs Challenger)
    run_step("python automate_mlops.py", "DÉCISION & PROMOTION")

    print("\n✅ PIPELINE TERMINÉ AVEC SUCCÈS")

if __name__ == "__main__":
    main()
