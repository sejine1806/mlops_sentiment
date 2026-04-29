import torch
import torch.nn as nn

class MambaSentimentModel(nn.Module):
    def __init__(self, vocab_size=5000, d_model=128, n_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba_core = nn.LSTM(d_model, d_model, batch_first=True) 
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.mamba_core(x)
        return self.classifier(x[:, -1, :])
