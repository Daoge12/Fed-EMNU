import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Machine Unlearning System")

    # 基础配置
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34"])
    parser.add_argument("--data", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])

    # 联邦学习参数
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--select_ratio", type=float, default=0.4)
    parser.add_argument("--global_epochs", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--iid", action="store_true", help="是否使用IID划分")

    # 遗忘参数
    parser.add_argument("--unlearn_class", type=int, nargs="+", required=True)
    parser.add_argument("--noise_copies", type=int, default=20,
                        help="噪声数据复制次数")
    parser.add_argument("--retain_ratio", type=float, default=0.1,
                        help="保留数据集比例")
    parser.add_argument("--impair_lr", type=float, default=0.02,
                        help="损害阶段学习率")
    parser.add_argument("--repair_lr"   , type=float, default=0.01,
                        help="修复阶段学习率")
    parser.add_argument("--client_id", type=int, required=True,
                        help="执行遗忘的客户端ID")
    parser.add_argument("--mode", type=str, default="noise",
                        choices=["noise", "retrain","ft", "ga"],
                        help="遗忘模式选择：noise(噪声注入)/retrain(重训练)/finetune(微调)/ga（梯度上升）")
    # 添加微调专用参数
    parser.add_argument("--retrain_epochs", type=int, default=50,
                        help="重训练轮数")
    parser.add_argument("--ft_lr", type=float, default=0.01,
                        help="微调学习率")
    parser.add_argument("--ft_epochs", type=int, default=5,
                        help="微调轮次")
    parser.add_argument("--ga_lr", type=float, default=0.001,
                        help="Gradient ascent learning rate")
    parser.add_argument("--ga_epochs", type=int, default=5,
                        help="Gradient ascent local epochs")
    return parser.parse_args()