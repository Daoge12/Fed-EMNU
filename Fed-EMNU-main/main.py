import random
import sys
import time

import numpy as np
import torch
from args import parse_args
from config import UNLEARN_CONFIG
from server import FederatedServer
from client import FederatedClient
from data_loader import load_datasets, split_data, split_dirichlet_non_iid
from utils import comprehensive_evaluation, analyze_data_distribution

class Logger(object): # log
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    args = parse_args()
    # 设置全局随机种子（确保所有随机过程一致）
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    sys.stdout = Logger(
        f"logs/{time.ctime()}{args.unlearn_class}--{args.mode} --{args.noise_copies} --{args.retain_ratio} --{args.iid}.txt")
    # 初始化联邦环境
    server = FederatedServer()
    train_set, _ = load_datasets()
    # 修改数据加载部分
    if args.iid:
        client_datasets = split_data(train_set, args.num_clients)
    else:
        client_datasets = split_dirichlet_non_iid(train_set, args.num_clients,alpha=0.5, seed=seed)

    analyze_data_distribution(client_datasets)

    # client_datasets = split_data(train_set, args.num_clients)
    clients = [FederatedClient(i, ds) for i, ds in enumerate(client_datasets)]

    # 联邦训练
    for epoch in range(args.global_epochs):
        selected = random.sample(clients, int(args.select_ratio * args.num_clients))
        updates = [client.local_train(server.get_global_model()) for client in selected]
        server.aggregate(updates)

        # 定期评估
        if (epoch + 1) % 5 == 0:
            metrics = comprehensive_evaluation(server.get_global_model())
            print(f"Epoch {epoch + 1:03d} | Test: {metrics['test_acc']:.2%} | "
                  f"Forget: {metrics['forget_acc']:.2%} | Retain: {metrics['retain_acc']:.2%}")

    # 执行遗忘
    start_time = time.perf_counter()  # 记录开始时间
    if args.mode =="noise":
        target_client = clients[args.client_id]
        package = server.prepare_unlearning_package(args.client_id, args.unlearn_class)
        unlearn_update = target_client.perform_unlearning(package)
        server.aggregate([unlearn_update])
    elif args.mode =="retrain":
        for client in clients:
            client.forget_classes = args.unlearn_class

        # 只针对目标客户端进行重训练
        for epoch in range(args.retrain_epochs):
            selected = random.sample(clients, int(args.select_ratio * args.num_clients))
            updates = [client.retain_train(server.get_global_model()) for client in selected]
            server.aggregate([updates])  # 直接应用单个客户端的更新

            # 输出重训练进度
            print(f"Retrain Epoch {epoch + 1}:")
            metrics = comprehensive_evaluation(server.get_global_model())
            print(f"Test Acc: {metrics['test_acc']:.2%} | "
                  f"Forget Acc: {metrics['forget_acc']:.2%} | "
                  f"Retain Acc: {metrics['retain_acc']:.2%}")
    elif args.mode == "ft":
        # 只设置目标客户端的遗忘类别
        target_client = clients[args.client_id]
        target_client.forget_classes = args.unlearn_class

        # 只针对目标客户端进行微调
        for epoch in range(args.ft_epochs):
            if epoch == 0:  # 第一轮全网络微调
                update = target_client.retain_train(server.get_global_model())
            else:  # 后续轮次仅微调顶层
                update = target_client.fine_tune(server.get_global_model())

            server.aggregate([update])

            # 评估
            metrics = comprehensive_evaluation(server.get_global_model())
            print(f"Fine-tune Epoch {epoch + 1} | Test: {metrics['test_acc']:.2%} | "
                  f"Forget: {metrics['forget_acc']:.2%} | Retain: {metrics['retain_acc']:.2%}")
    elif args.mode == "ga":
        # 只设置目标客户端的遗忘类别
        target_client = clients[args.client_id]
        target_client.forget_classes = args.unlearn_class

        # 只针对目标客户端进行梯度上升
        for epoch in range(args.ga_epochs):
            update = target_client.gradient_ascent_unlearn(server.get_global_model())
            server.aggregate([update])

        # # 可选修复阶段（使用保留数据）
        # for epoch in range(args.ga_epochs):
        #     repair_update = target_client.retain_train(server.get_global_model())
        #     server.aggregate([repair_update])

            # 评估
            metrics = comprehensive_evaluation(server.get_global_model())
            print(f"GA Repair Epoch {epoch + 1} | Test: {metrics['test_acc']:.2%} | "
                  f"Forget: {metrics['forget_acc']:.2%} | Retain: {metrics['retain_acc']:.2%}")

    end_time = time.perf_counter()  # 记录结束时间
    unlearn_duration = end_time - start_time  # 计算耗时
    print(f"\nUnlearning Time: {unlearn_duration:.4f} seconds")  # 输出到日志

    # 最终评估
    final_metrics = comprehensive_evaluation(server.get_global_model())
    print("\n=== Final Results ===")
    print(f"Test Accuracy: {final_metrics['test_acc']:.2%}")
    print(f"Forget Class Accuracy: {final_metrics['forget_acc']:.2%}")
    print(f"Retain Class Accuracy: {final_metrics['retain_acc']:.2%}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()