import torch
from args import parse_args

args = parse_args()

# 设备配置
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# 联邦学习配置
FEDERATION_CONFIG = {
    "num_clients": args.num_clients,
    "select_ratio": args.select_ratio,
    "global_epochs": args.global_epochs,
    "local_epochs": args.local_epochs,
    "batch_size": args.batch_size,
    "lr": args.lr
}

# 遗忘配置
UNLEARN_CONFIG = {
    "target_classes": args.unlearn_class,
    "noise_copies": args.noise_copies,
    "retain_ratio": args.retain_ratio,
    "impair_lr": args.impair_lr,
    "repair_lr": args.repair_lr,
    "noise_epochs": 10,
    "noise_steps": 8,
    "noise_lr": 0.1,
    "batch_size": args.batch_size,
    "repair_epochs": 3,
    "finetune_lr": args.ft_lr,
    "finetune_epochs": args.ft_epochs,
    "ga_lr": args.ga_lr,
    "ga_epochs": args.ga_epochs}

# 模型配置
MODEL_CONFIG = {
    "name": args.model,
    "num_classes": 10 if args.data == "cifar10" else 100
}