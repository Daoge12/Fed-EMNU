import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_datasets():
    """加载标准化数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10("./data", train=False, transform=transform)
    return train_set, test_set


def split_data(dataset, num_clients):
    """IID数据划分"""
    indices = np.random.permutation(len(dataset))
    return [Subset(dataset, indices[i::num_clients]) for i in range(num_clients)]


# def split_dirichlet_non_iid(dataset, num_clients, alpha):
#     """Dirichlet分布非IID划分"""
#     # 按类别组织数据索引
#     class_indices = {i: [] for i in range(10)}  # CIFAR-10有10个类
#     for idx, (_, label) in enumerate(dataset):
#         class_indices[label].append(idx)
#
#     # 为每个客户端生成类别分布
#     client_distributions = np.random.dirichlet(
#         np.repeat(alpha, 10),  # 每个类使用相同alpha
#         size=num_clients
#     )
#
#     client_subsets = []
#     for client_id in range(num_clients):
#         subset_indices = []
#         for class_id in range(10):
#             # 计算该类分配给该客户端的样本数量
#             total = len(class_indices[class_id])
#             proportion = client_distributions[client_id, class_id]
#             num_samples = int(proportion * total)
#
#             # 随机抽取样本（无放回）
#             selected = np.random.choice(
#                 class_indices[class_id],
#                 size=num_samples,
#                 replace=False
#             )
#             subset_indices.extend(selected)
#
#             # 从总池中移除已选样本
#             class_indices[class_id] = list(
#                 set(class_indices[class_id]) - set(selected)
#             )
#
#         client_subsets.append(Subset(dataset, subset_indices))
#
#     return client_subsets
def split_dirichlet_non_iid(dataset, num_clients, alpha,seed=42):
    """Dirichlet分布非IID划分"""
    # 设置随机种子
    np.random.seed(seed)
    # 按类别组织数据索引
    class_indices = {i: [] for i in range(10)}  # CIFAR-10有10个类
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 初始化客户端索引列表
    client_subsets = [[] for _ in range(num_clients)]

    # 按类别独立分配
    for class_id in range(10):
        # 步骤1：获取当前类别的所有样本索引
        class_samples = class_indices[class_id]
        num_samples = len(class_samples)

        # 步骤2：生成该类别的客户端分配比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()  # 确保概率和为1

        # 步骤3：计算每个客户端分配的样本数（含剩余样本处理）
        allocations = (proportions * num_samples).astype(int)
        remaining = num_samples - allocations.sum()

        # 将剩余样本分配给前remaining个客户端
        allocations[:remaining] += 1

        # 步骤4：打乱样本并分配
        np.random.shuffle(class_samples)
        ptr = 0
        for client_id in range(num_clients):
            client_size = allocations[client_id]
            client_subsets[client_id].extend(class_samples[ptr:ptr + client_size])
            ptr += client_size

    # 创建Subset对象
    return [Subset(dataset, indices) for indices in client_subsets]


def get_test_loaders(target_classes):
    """获取三类测试数据加载器"""
    _, test_set = load_datasets()

    # 转换张量标签为Python整数
    def get_label(idx):
        return test_set[idx][1]

    # 构建索引
    forget_idx = [i for i in range(len(test_set)) if get_label(i) in target_classes]
    retain_idx = [i for i in range(len(test_set)) if get_label(i) not in target_classes]

    return (
        DataLoader(test_set, batch_size=256),  # 完整测试集
        DataLoader(Subset(test_set, forget_idx), batch_size=256),  # 遗忘类
        DataLoader(Subset(test_set, retain_idx), batch_size=256)  # 保留类
    )