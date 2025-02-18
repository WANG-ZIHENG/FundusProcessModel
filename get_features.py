# get_features.py
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
from Models import Model
import numpy as np
class Centers():
    def __init__(self,training_dataset,args,model,device, batch_size=32,alpha=0.5):
        print("init Centers")

        self.class_centers = self.extract_class_centers(training_dataset,model=model, batch_size=batch_size,)

        self.class_centers = {k:v.to(args.device) for k,v in self.class_centers.items()}

        self.args = args
        self.alpha = alpha
    def extract_class_centers(self,dataset,model, batch_size=32,):




        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset.transform = transform
        # 加载数据集
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        # 使用GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 提取特征
        model.eval()
        features_per_class = defaultdict(list)
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                features,_,_ = model(input_imgs=inputs,cal_center=True)
                for output, label in zip(features.cpu(), labels):
                    features_per_class[label.item()].append(output)

        # 计算每个类别的特征均值
        class_centers = {label: torch.mean(torch.stack(features), dim=0)
                         for label, features in features_per_class.items()}

        class_centers = {label: center
                         for label, center in class_centers.items()}

        return class_centers

    def update_alpha(self,epoch):
        self.alpha = 0.25 * (1. + math.cos(math.pi * (epoch + 1) / (self.args.max_epochs + 1)))
        if self.alpha < 0.05:
            self.alpha = 0.05
    def update_class_centers(self,epoch, features, labels):
        """
        update class center
        更新类中心。
        :param class_centers: now class centers dict
        :param features: batch features
        :param labels:  batch labels
        :param alpha: updating ratio
        :return: updated class centers dict
        """

        self.update_alpha(epoch)

        for label in self.class_centers.keys():  # 使用字典的键进行迭代
            # class feature with specific label
            class_features = features[labels == label]
            # if there is feature in the mini-batch, update the class center
            if class_features.size(0) > 0:
                new_centers = torch.zeros_like(self.class_centers[label])
                for i in range(features.shape[1]):
                    f_i = class_features[:,i,:]
                    f_i = f_i.mean(dim=0)
                    # update class center
                    new_centers += (1 - self.alpha) * self.class_centers[label] + self.alpha *f_i
                new_centers = new_centers/features.shape[1]

                self.class_centers[label] =new_centers
        return self.class_centers

    def old_update_class_centers(self,epoch, features, labels):
        """
        update class center
        更新类中心。
        :param class_centers: now class centers dict
        :param features: batch features
        :param labels:  batch labels
        :param alpha: updating ratio
        :return: updated class centers dict
        """

        self.update_alpha(epoch)

        for label in self.class_centers.keys():  # 使用字典的键进行迭代
            # class feature with specific label
            class_features = features[labels == label]
            # if there is feature in the mini-batch, update the class center
            if class_features.size(0) > 0:
                new_center = class_features.mean(dim=0)
                # update class center
                self.class_centers[label] = (1 - self.alpha) * self.class_centers[label] + self.alpha *new_center
        return self.class_centers
