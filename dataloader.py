import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random

from torchvision import transforms

random.seed(1143)

def populate_pair_list(root_path, lowlight_subdir="low", highlight_subdir="high"):

    lowlight_dir = os.path.join(root_path, lowlight_subdir)
    highlight_dir = os.path.join(root_path, highlight_subdir)

    if not os.path.isdir(lowlight_dir):
        raise FileNotFoundError(f"低光照图像目录未找到: {lowlight_dir}")
    if not os.path.isdir(highlight_dir):
        raise FileNotFoundError(f"高光照（目标）图像目录未找到: {highlight_dir}")

    # 获取低光照图像的所有路径
    lowlight_image_paths = sorted(glob.glob(os.path.join(lowlight_dir, "*.png")))

    paired_list = []
    for low_path in lowlight_image_paths:
        filename = os.path.basename(low_path) # 提取文件名，用于匹配
        high_path = os.path.join(highlight_dir, filename) # 构建对应的高光照图像路径
        if os.path.exists(high_path):
            paired_list.append((low_path, high_path))
        else:
            print(f"警告: 未找到低光照图像 {low_path} 对应的目标高光照图像。")

    if not paired_list:
        raise RuntimeError(f"在 {root_path} 及其子目录 '{lowlight_subdir}' 和 '{highlight_subdir}' 中未找到任何图像对。请检查您的数据结构和文件扩展名。")

    random.shuffle(paired_list) # 随机打乱图像对
    return paired_list


class lowlight_loader(data.Dataset):
    def __init__(self, data_root_path, lowlight_subdir="low", highlight_subdir="high"):
        # 调用新的函数来获取图像对列表
        self.paired_list = populate_pair_list(data_root_path, lowlight_subdir, highlight_subdir)
        self.size = 256 # 图像将被resize到的尺寸
        self.data_list = self.paired_list # 现在这是一个 (low_path, high_path) 元组的列表
        print("训练样本总数 (图像对):", len(self.paired_list))

        # 定义图像转换：PIL Image -> Tensor，并归一化到 [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        lowlight_path, highlight_path = self.data_list[index]

        # 打开图像并确保为 RGB 格式
        data_lowlight = Image.open(lowlight_path).convert("RGB")
        data_highlight = Image.open(highlight_path).convert("RGB")

        # Resize 图像
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.Resampling.LANCZOS)
        data_highlight = data_highlight.resize((self.size, self.size), Image.Resampling.LANCZOS)

        # 应用转换
        data_lowlight = self.transform(data_lowlight)
        data_highlight = self.transform(data_highlight)

        # 返回低光照图像和高光照图像（目标图像）
        return data_lowlight, data_highlight

    def __len__(self):
        return len(self.data_list)