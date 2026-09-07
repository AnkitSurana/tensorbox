"""
PyTorch Two-Tower Recommendation Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users=1000, num_items=500, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, user_ids, item_ids):
        u_emb = F.normalize(self.user_mlp(self.user_embedding(user_ids)), p=2, dim=-1)
        i_emb = F.normalize(self.item_mlp(self.item_embedding(item_ids)), p=2, dim=-1)
        return (u_emb * i_emb).sum(dim=-1)
