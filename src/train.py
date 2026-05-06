import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from datasets import load_dataset
from transformers import AutoTokenizer
from mamba_arch import MambaSentimentModel

def train():
    mlflow.set_experiment("Mamba_Real_Training")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    with mlflow.start_run():
        # 1. On augmente un peu la donnée pour de meilleurs résultats
        print("📥 Chargement du dataset IMDB (2000 exemples)...")
       # dataset = load_dataset("imdb", split="train[:5000]")
        full_dataset = load_dataset("imdb", split="train")
        dataset = full_dataset.shuffle(seed=42).select(range(5000))

        model = MambaSentimentModel(vocab_size=30522, d_model=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        print("🚀 Entraînement en cours...")
        model.train()
        
        # 2. Boucle sur 10 époques
        for epoch in range(5):
            epoch_loss = 0
            # 3. On parcourt TOUT le dataset (pas juste 10 lignes)
            for i in range(len(dataset)):
                example = dataset[i]
                inputs = tokenizer(example['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=64)
                label = torch.tensor([example['label']])

                optimizer.zero_grad()
                output = model(inputs['input_ids'])
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            # 4. On envoie la moyenne de l'erreur à MLflow pour voir la courbe
            avg_loss = epoch_loss / len(dataset)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch} terminée - Loss moyenne: {avg_loss:.4f}")

        # Sauvegarde finale
        mlflow.pytorch.log_model(model, "mamba_model")
        torch.save(model.state_dict(), "src/mamba_weights.pt")
        print("✅ Poids du modèle sauvegardés dans src/mamba_weights.pt")

if __name__ == "__main__":
    train()
