import torch
from data_loader import get_test_loaders
from config import DEVICE, UNLEARN_CONFIG
from collections import namedtuple
import numpy as np

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def comprehensive_evaluation(model):
    target_classes = UNLEARN_CONFIG["target_classes"]
    full_loader, forget_loader, retain_loader = get_test_loaders(target_classes)
    return {
        "test_acc": evaluate_model(model, full_loader),
        "forget_acc": evaluate_model(model, forget_loader),
        "retain_acc": evaluate_model(model, retain_loader)
    }

def analyze_data_distribution(clients_datasets):
    """分析各客户端数据分布"""
    for cid, subset in enumerate(clients_datasets):
        class_count = {i:0 for i in range(10)}
        for _, label in subset:
            class_count[label] += 1
        print(f"Client {cid}:")
        print(" | ".join([f"Class {k}:{v}" for k,v in class_count.items()]))