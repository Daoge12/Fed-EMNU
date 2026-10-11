import copy
import itertools
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, TensorDataset, ConcatDataset

from args import parse_args
from config import DEVICE, UNLEARN_CONFIG, FEDERATION_CONFIG
from model import GlobalModel
import server
from utils import evaluate_model

args = parse_args()
class FederatedClient:
    def __init__(self, client_id, dataset):
        self.global_model = GlobalModel().to(DEVICE)
        self.id = client_id
        self.dataset = dataset
        self.model = None
        self.retain_sub_loader = None
        self.forget_classes = None  # 记录需要遗忘的类别
        self.ft_optimizer = None


    def _compute_confidence(self, model):
        """计算并存储样本置信度"""
        model.eval()
        all_confidences = []

        # 遍历整个本地数据集
        loader = DataLoader(self.dataset, batch_size=FEDERATION_CONFIG["batch_size"])
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probabilities, dim=1)
                all_confidences.extend(max_probs.cpu().tolist())

        # 存储原始索引到置信度的映射
        self.confidence_scores = {i: conf for i, conf in enumerate(all_confidences)}

    def local_train(self, global_model):
        """本地训练方法"""
        # 初始化本地模型
        self.model = copy.deepcopy(global_model).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=FEDERATION_CONFIG["lr"])
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=FEDERATION_CONFIG["lr"],momentum=0.9,weight_decay=0.0005)

        # 创建数据加载器
        train_loader = DataLoader(
            self.dataset,
            batch_size=FEDERATION_CONFIG["batch_size"],
            shuffle=True
        )

        # 训练循环
        self.model.train()
        for _ in range(FEDERATION_CONFIG["local_epochs"]):
            for inputs, labels in train_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
        # 在训练结束后计算置信度（新增）
        self._compute_confidence(self.model)
        return self.model.state_dict()
    def prepare_retain_data(self, forget_classes):
        if args.iid:
            print("iid")
            """准备保留数据集"""
            retain_indices = [
                i for i, (_, label) in enumerate(self.dataset)
                if label not in forget_classes
            ]
            retain_subset = Subset(self.dataset, retain_indices)

            # 按比例采样
            sampler = RandomSampler(
                retain_subset,
                num_samples=int(len(retain_subset) * UNLEARN_CONFIG["retain_ratio"])
            )
            self.retain_sub_loader = DataLoader(
                retain_subset,
                batch_size=UNLEARN_CONFIG["batch_size"],
                sampler=sampler
            )
            return self.retain_sub_loader
        else:
            """按类别独立筛选高置信度样本，并处理样本不足"""
            # Step 1: 组织保留类别的索引
            class_indices = {}
            for idx, (_, label) in enumerate(self.dataset):
                if label not in forget_classes:
                    if label not in class_indices:
                        class_indices[label] = []
                    class_indices[label].append(idx)

            # Step 2: 按类别独立排序和采样
            class_retain_indices = {}  # 记录每个类别的保留索引
            for cls, indices in class_indices.items():
                # 处理当前类别
                cls_confidences = [(i, self.confidence_scores[i]) for i in indices]
                sorted_items = sorted(
                    cls_confidences,
                    key=lambda x: x[1],
                    reverse=True
                )
                cls_retain_num = int(len(sorted_items) * UNLEARN_CONFIG["retain_ratio"])
                selected = [x[0] for x in sorted_items[:cls_retain_num]]
                class_retain_indices[cls] = selected

            # Step 3: 处理样本不足的类别（新增逻辑）
            for cls in class_retain_indices:
                selected = class_retain_indices[cls]
                num_selected = len(selected)
                if 0 < num_selected < 10:
                    required_copies = (10 // num_selected) + 1  # 至少复制到10个
                    class_retain_indices[cls] = selected * required_copies

            # 重新构建最终索引列表
            retain_indices = list(itertools.chain.from_iterable(class_retain_indices.values()))

            # Step 4: 构建数据加载器
            retain_subset = Subset(self.dataset, retain_indices)
            self.retain_sub_loader = DataLoader(
                retain_subset,
                batch_size=UNLEARN_CONFIG["batch_size"],
                shuffle=True,
                drop_last=True
            )
            return self.retain_sub_loader


    def perform_unlearning(self, package):
        """执行本地遗忘"""
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(package["model_state"])
        model.to(DEVICE)

        self.prepare_retain_data(package["target_classes"])
        print(evaluate_model(model, self.retain_sub_loader))
        # # 损害阶段
        # for noise in package["noises"]:
        #     noise = noise.to(DEVICE)
        #     self._impair_step(model, noise)
        # 损害阶段：遍历每个噪声及其对应类
        for noise, target_class in zip(package["noises"], package["target_classes"]):
            noise = noise.to(DEVICE)
            self._impair_step(model, noise, target_class)  # 传入目标类
        print(evaluate_model(model, self.retain_sub_loader))
        # 修复阶段
        for i in range(UNLEARN_CONFIG["repair_epochs"]):
            self._repair_step(model)
            print(evaluate_model(model, self.retain_sub_loader))
        return model.state_dict()

    def _impair_step(self, model, noise, target_class):
        """为每个噪声绑定目标类标签"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=UNLEARN_CONFIG["impair_lr"])

        # 生成对应类别的标签
        noise_labels = torch.full(
            (noise.size(0),),
            target_class,  # 使用传入的目标类
            device=DEVICE,
            dtype=torch.long
        )

        # 复制噪声并与保留数据混合
        repeated_noise = noise.repeat(UNLEARN_CONFIG['noise_copies'], 1, 1, 1)
        repeated_labels = noise_labels.repeat(UNLEARN_CONFIG['noise_copies'])

        noise_dataset = TensorDataset(repeated_noise, repeated_labels)
        retain_dataset = self.retain_sub_loader.dataset
        noise_loader = DataLoader(noise_dataset, batch_size=UNLEARN_CONFIG['batch_size'])
        mixed_dataset = ConcatDataset([noise_dataset, retain_dataset])
        mixed_loader = DataLoader(mixed_dataset, batch_size=UNLEARN_CONFIG['batch_size'])

        # 训练过程
        for inputs, labels in mixed_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
        print("noise", evaluate_model(model, noise_loader))

    def _repair_step(self, model):
        """数据修复"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=UNLEARN_CONFIG["repair_lr"])
        for inputs, labels in self.retain_sub_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

    def _get_full_retain_loader(self, forget_classes):
        """获取完整的保留数据集加载器"""
        # 生成保留样本索引
        retain_indices = [
            idx for idx, (_, label) in enumerate(self.dataset)
            if label not in forget_classes
        ]
        retain_subset = Subset(self.dataset, retain_indices)

        return DataLoader(
            retain_subset,
            batch_size=UNLEARN_CONFIG["batch_size"],
            shuffle=True,
            drop_last=False
        )

    def retain_train(self, global_model):
        """使用完整保留数据集重新训练"""
        self.model = copy.deepcopy(global_model).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=FEDERATION_CONFIG["lr"])

        # 获取完整数据集加载器
        retain_loader = self._get_full_retain_loader(self.forget_classes)

        self.model.train()
        for _ in range(FEDERATION_CONFIG["local_epochs"]):
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()

    def fine_tune(self, global_model):
        """基于完整保留数据的微调"""
        self.model = copy.deepcopy(global_model).to(DEVICE)

        self.ft_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=UNLEARN_CONFIG["finetune_lr"],
            # momentum=0.9,
            # weight_decay=5e-4
        )

        # 获取完整数据集加载器
        retain_loader = self._get_full_retain_loader(self.forget_classes)

        self.model.train()
        for _ in range(UNLEARN_CONFIG["finetune_epochs"]):
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                self.ft_optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()

                # 冻结非全连接层
                for name, param in self.model.named_parameters():
                    if 'fc' not in name:  # 假设最后一层是fc层
                        param.grad = None

                self.ft_optimizer.step()
        return self.model.state_dict()

    def gradient_ascent_unlearn(self, global_model):
        """梯度上升遗忘方法"""
        self.model = copy.deepcopy(global_model).to(DEVICE)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=UNLEARN_CONFIG["ga_lr"],
            # momentum=0.9,
            # weight_decay=0
        )

        # 获取遗忘数据集（从本地数据筛选目标类）
        forget_loader = self._get_forget_data_loader()

        # 梯度上升训练
        self.model.train()
        for _ in range(UNLEARN_CONFIG["ga_epochs"]):
            for inputs, labels in forget_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                (-loss).backward()  # 关键：梯度反转
                optimizer.step()

        return self.model.state_dict()

    def _get_forget_data_loader(self):
        """生成遗忘类数据加载器"""
        # 筛选目标类样本
        forget_indices = [
            idx for idx, (_, label) in enumerate(self.dataset)
            if label in self.forget_classes
        ]
        if len(forget_indices) < 10:
            copies = 10 // max(1, len(forget_indices)) + 1
            forget_indices = forget_indices * copies

        forget_subset = Subset(self.dataset, forget_indices)

        return DataLoader(
            forget_subset,
            batch_size=UNLEARN_CONFIG["batch_size"],
            shuffle=True,
            drop_last=True
        )