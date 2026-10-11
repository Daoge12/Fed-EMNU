import copy
import torch
from model import GlobalModel
from noise_generator import ClassWiseNoiseGenerator
from config import DEVICE, UNLEARN_CONFIG


class FederatedServer:
    def __init__(self):
        self.global_model = GlobalModel().to(DEVICE)
        self.noise_generators = {}

    def aggregate(self, client_updates):
        """联邦平均聚合"""
        global_state = self.global_model.state_dict()
        for key in global_state:
            global_state[key] = torch.stack(
                [update[key].float() for update in client_updates]
            ).mean(dim=0)
        self.global_model.load_state_dict(global_state)

    def prepare_unlearning_package(self, client_id, target_classes):
        # """准备遗忘数据包"""
        # if client_id not in self.noise_generators:
        #     self.noise_generators[client_id] = [
        #         ClassWiseNoiseGenerator(cls) for cls in target_classes
        #     ]
        #     for gen in self.noise_generators[client_id]:
        #         gen.optimize(self.global_model)
        #
        # return {
        #     "model_state": copy.deepcopy(self.global_model.state_dict()),
        #     "noises": [gen.noise for gen in self.noise_generators[client_id]],
        #     "target_classes": target_classes
        # }
        """为每个遗忘类生成独立噪声"""
        if client_id not in self.noise_generators:
            # 为每个目标类创建独立的噪声生成器
            self.noise_generators[client_id] = {}
            for cls in target_classes:
                generator = ClassWiseNoiseGenerator(cls)
                generator.optimize(self.global_model)
                self.noise_generators[client_id][cls] = generator

        # 收集所有噪声和对应标签
        noises = []
        for cls in target_classes:
            noises.append(self.noise_generators[client_id][cls].noise)

        return {
            "model_state": copy.deepcopy(self.global_model.state_dict()),
            "noises": noises,  # 包含多个类的噪声
            "target_classes": target_classes  # 明确传递目标类列表
        }

    def get_global_model(self):
        return copy.deepcopy(self.global_model)