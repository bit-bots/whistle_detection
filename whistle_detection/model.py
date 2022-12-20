import torch.nn as nn
from torchvision import models


def get_model(device):
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)
