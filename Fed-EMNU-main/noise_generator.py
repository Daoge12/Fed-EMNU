import torch
import torch.nn as nn
from config import DEVICE, UNLEARN_CONFIG
import torch.nn.functional as F

class ClassWiseNoiseGenerator(nn.Module):
    """类感知噪声生成器"""

    def __init__(self, target_class):
        super().__init__()
        self.target_class = target_class
        self.noise = nn.Parameter(
            torch.randn(UNLEARN_CONFIG["batch_size"], 3, 32, 32,device=DEVICE),
            requires_grad=True
        )

    # def optimize(self, model):
    #     """误差最大化优化"""
    #     model.eval()
    #     print(self.noise.is_leaf)  # 应该输出 True
    #     optimizer = torch.optim.Adam([self.noise], lr=UNLEARN_CONFIG["noise_lr"])
    #
    #     for _ in range(UNLEARN_CONFIG["noise_epochs"]):
    #         for _ in range(UNLEARN_CONFIG["noise_steps"]):
    #             optimizer.zero_grad()
    #             labels = torch.full(
    #                 (UNLEARN_CONFIG["batch_size"],),
    #                 self.target_class,
    #                 device=DEVICE
    #             )
    #             outputs = model(self.noise)
    #             # loss = -nn.CrossEntropyLoss()(outputs, labels) + 0.1 * torch.mean(self.noise ** 2)
    #             loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(self.noise), [1, 2, 3]))
    #             loss.backward()
    #             optimizer.step()
    #         print("Loss: {}".format(loss.item()))
    #     return self.noise.detach()
    def optimize(self, model):
        """确保噪声优化对应目标类"""
        model.eval()
        optimizer = torch.optim.Adam([self.noise], lr=UNLEARN_CONFIG["noise_lr"])
        # optimizer = torch.optim.SGD([self.noise], lr=UNLEARN_CONFIG["noise_lr"], momentum=0.9,
        #                             weight_decay=0.0005)

        for _ in range(UNLEARN_CONFIG["noise_epochs"]):
            for _ in range(UNLEARN_CONFIG["noise_steps"]):
                optimizer.zero_grad()
                # 标签必须与初始化时的target_class一致
                labels = torch.full(
                    (UNLEARN_CONFIG["batch_size"],),
                    self.target_class,  # 使用类初始化时存储的目标类
                    device=DEVICE
                )
                outputs = model(self.noise)
                loss = -F.cross_entropy(outputs, labels) + 0.1 * torch.norm(self.noise)
                loss.backward()
                optimizer.step()
            # print("Loss: {}".format(loss.item()))
        return self.noise.detach()