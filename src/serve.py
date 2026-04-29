from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from transformers import AutoTokenizer
from mamba_arch import MambaSentimentModel

weights_path = "src/mamba_weights.pt"
app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MambaSentimentModel(vocab_size=30522, d_model=128)

if os.path.exists(weights_path):
    print(f"🧠 Chargement des poids depuis {weights_path}...")
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("✅ SUCCÈS : Le modèle entraîné est opérationnel !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
else:
    # On affiche le dossier actuel pour comprendre pourquoi il ne le voit pas
    print(f"❌ ERREUR : {weights_path} introuvable.")
    print(f"📁 Dossier actuel : {os.getcwd()}")
    print(f"📂 Contenu : {os.listdir('.')}")

model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    # On transforme TON texte avec le tokenizer
    inputs = tokenizer(request.text, return_tensors="pt", padding='max_length', truncation=True, max_length=50)
    
    with torch.no_grad():
        output = model(inputs['input_ids'])
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    
    return {
        "text": request.text,
        "sentiment": "Positif" if pred == 1 else "Négatif",
        "confidence": f"{torch.max(prob).item():.2%}"
    }


