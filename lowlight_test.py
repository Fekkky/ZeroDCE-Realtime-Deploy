import torch
import torchvision
import torch.optim
import os
import sys
import argparse
import time
import dataloader # 假设这是用于训练的，在此推理中未直接使用
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

# 获取当前脚本所在目录 (Zero-DCE_code)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_environment():
    """检查运行环境并输出调试信息"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


def print_directory_contents(path, depth=0):
    """递归打印目录内容，用于调试"""
    indent = "  " * depth
    print(f"{indent}[D] {path}")

    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print_directory_contents(item_path, depth + 1)
            else:
                print(f"{indent}  [F] {item}")
    except Exception as e:
        print(f"{indent}  [!] Error accessing {path}: {str(e)}")


def find_test_images():
    """查找所有测试图像，支持多种格式"""
    test_data_path = os.path.join(PROJECT_ROOT, 'data/test_data')
    print(f"Searching for images in: {test_data_path}")

    # 打印数据目录内容用于调试
    print("Contents of data directory:")
    print_directory_contents(os.path.join(PROJECT_ROOT, 'data'))

    # 支持常见图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    test_images = []

    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_data_path, '**', ext), recursive=True))

    return test_images


def check_paths():
    """检查关键文件和目录是否存在"""
    test_data_path = os.path.join(PROJECT_ROOT, 'data/test_data')
    # 根据 lowlight 函数中使用的模型路径，将此处检查的模型路径也改为 'best_model_1.pth'
    model_path = os.path.join(PROJECT_ROOT, 'snapshots/best_model_best_3.pth')

    print(f"Checking test data path: {test_data_path}")
    if not os.path.exists(test_data_path):
        print(f"Error: Test data path '{test_data_path}' does not exist!")
        return False

    test_images = find_test_images()
    if not test_images:
        print(f"Error: No test images found in '{test_data_path}'!")
        return False
    else:
        print(f"Found {len(test_images)} test images")

    print(f"Checking model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist!")
        return False

    return test_images


def lowlight(image_path):
    """处理单张低光照图像并增强"""
    print(f"\nProcessing image: {image_path}")

    # 确保使用绝对路径
    if not os.path.isabs(image_path):
        image_path = os.path.join(PROJECT_ROOT, image_path)

    # 检查输入文件是否存在
    if not os.path.exists(image_path):
        print(f"Warning: Image '{image_path}' does not exist, skipping!")
        return

    try:
        # 设置 CUDA 设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- 修改部分开始 ---
        # 加载图像为 PIL Image，并转换为 RGB 格式，以确保一致性
        data_lowlight_pil = Image.open(image_path).convert("RGB")

        # 定义图像变换管道，包括缩放和转换为张量
        transform = transforms.Compose([
            transforms.Resize((400, 600)), # 缩放图像为 (高度, 宽度) = (400, 600)
            transforms.ToTensor()          # 将 PIL Image 转换为 Tensor，并自动归一化到 [0, 1] 范围
                                           # 同时将维度从 H, W, C 调整为 C, H, W
        ])

        # 应用变换
        data_lowlight = transform(data_lowlight_pil)
        # --- 修改部分结束 ---

        # 将数据移动到设备，并添加批次维度
        data_lowlight = data_lowlight.to(device).unsqueeze(0)

        # 加载模型
        model_path = os.path.join(PROJECT_ROOT, 'snapshots/best_model_best_3.pth')
        DCE_net = model.enhance_net_nopool().to(device)
        DCE_net.load_state_dict(torch.load(model_path, map_location=device))
        DCE_net.eval()  # 设置为评估模式

        # 处理图像
        print("Starting image enhancement...")
        start = time.time()
        with torch.no_grad():
            # 模型输出三个张量，我们关注第二个 (增强后的图像)
            _, enhanced_image, _ = DCE_net(data_lowlight)
        end_time = time.time() - start
        print(f"Enhancement completed in {end_time:.4f} seconds")

        # 准备保存结果
        # 将输出保存到名为 'result_resized' 的新目录中
        result_path = image_path.replace('test_data', 'result_best_3')
        result_dir = os.path.dirname(result_path)

        # 创建结果目录（如果不存在）
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print(f"Created directory: {result_dir}")

        # 保存结果
        torchvision.utils.save_image(enhanced_image, result_path)
        print(f"Enhanced image saved to: {result_path}")

    except Exception as e:
        print(f"Error processing image '{image_path}': {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈信息


if __name__ == '__main__':
    print("Starting low-light image enhancement script...")
    print(f"Project root directory: {PROJECT_ROOT}")

    # 检查环境
    check_environment()

    # 检查路径并获取测试图像
    test_images = check_paths()
    if not test_images:
        sys.exit(1)

    # 处理测试图像
    with torch.no_grad():
        print(f"Processing {len(test_images)} images...")

        for i, image_path in enumerate(test_images):
            print(f"\nImage {i + 1}/{len(test_images)}")
            lowlight(image_path)

    print("\nScript completed!")