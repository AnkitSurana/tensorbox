"""
Tests for Two-Tower Recommender.
"""

import torch
from model import TwoTowerModel

def test_two_tower_forward():
    model = TwoTowerModel(num_users=100, num_items=50, embedding_dim=16)
    users = torch.tensor([0, 5, 10])
    items = torch.tensor([1, 8, 20])
    scores = model(users, items)
    assert scores.shape == (3,)
    assert (scores >= -1.0).all() and (scores <= 1.0).all()
