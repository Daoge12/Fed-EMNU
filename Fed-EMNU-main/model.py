import torch.nn as nn
from torchvision.models import resnet18, resnet34
from config import MODEL_CONFIG


class GlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        if MODEL_CONFIG["name"] == "resnet18":
            self.base = resnet18(num_classes=MODEL_CONFIG["num_classes"])
        elif MODEL_CONFIG["name"] == "resnet34":
            self.base = resnet34(num_classes=MODEL_CONFIG["num_classes"])

    def forward(self, x):
        return self.base(x)

    def get_state(self):
        return self.state_dict()

    def load_state(self, state_dict):
        self.load_state_dict(state_dict)