import torch
import torch.nn as nn
import torch.nn.functional as F


class URLClassifier(nn.Module):
    def __init__(
        self, input_dim, embedding_dim=128, hidden_dim=64, output_dim=3, dropout=0.5
    ):
        super(URLClassifier, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Classification layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)

        x = torch.mean(x, dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
