# ZeroDCE-Realtime-Deploy

基于 Zero-DCE 算法的实时低光照图像增强系统，支持摄像头实时处理与 FastAPI 服务部署。

---

## 项目简介

本项目复现了 CVPR 2020 论文 *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*，并在此基础上进行工程化落地：

- 无监督训练，不依赖成对的低光/正常光图像
- 扩展支持实时摄像头视频流处理，平均处理速度达到 **45 FPS**
- 封装为 FastAPI 推理服务，提供标准化 HTTP 接口
- 导出 ONNX 格式，对比不同部署方式的推理性能

---


## 核心算法

Zero-DCE 将图像增强问题转化为**像素级曲线参数估计**，通过轻量网络 DCE-Net 为每张图片生成专属的增强曲线：

```
LE(x, α) = x + α · x · (1 - x)
```

训练采用四个无参考损失函数，无需成对训练数据：

- **空间一致性损失**：保持增强前后邻域像素的亮度差异关系
- **曝光控制损失**：约束局部区域平均亮度趋近目标值（0.6）
- **颜色恒常损失**：基于灰度世界理论，防止增强后色偏
- **光照平滑损失**：对曲线参数施加 TV 正则，避免空间伪影

---

## 环境依赖

```
Python 3.10.0
torch 2.1.1+cu118
torchvision
fastapi
uvicorn
python-multipart
opencv-python
Pillow
onnx
onnxruntime-gpu
seaborn
matplotlib
torchmetrics
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 数据集

使用 [LOL Dataset](https://daooshee.github.io/BMVC2018website/) 进行训练和测试。

目录结构：

```
data/
  train_data/        # 训练集（仅低光照图像）
  test_data/
    low/             # 测试集低光照图像
    high/            # 测试集正常光图像（用于评估指标）
```

---

## 训练

```bash
python lowlight_train.py \
  --data_root_path data/train_data \
  --num_epochs 200 \
  --train_batch_size 4 \
  --lr 0.0001
```

训练完成后模型权重保存至 `snapshots/`，训练曲线图保存至 `visualization/`。

---

## 测试

```bash
python lowlight_test.py
```

增强结果保存至 `data/result/`。

---

## 实时摄像头增强

```bash
python camera_enhance.py
```

启动后弹出窗口，左侧为原始摄像头画面，右侧为实时增强结果，左上角显示实时 FPS。按 `Q` 退出。

**实测性能：平均 45 FPS**

---

## FastAPI 服务部署

启动服务：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/docs` 打开交互式文档，直接上传图片测试。

接口说明：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/enhance` | 上传低光照图片，返回增强结果 |
| GET | `/health` | 检查服务状态 |

---

## ONNX 导出

```bash
python export_onnx.py
```

导出模型至 `snapshots/dce_net.onnx`，并自动对比 PyTorch 与 ONNX Runtime 的推理速度。

---

## 项目结构

```
ZeroDCE-Realtime-Enhancement/
├── data/                  # 数据集
├── snapshots/             # 模型权重
├── visualization/         # 训练曲线图
├── model.py               # DCE-Net 网络结构
├── Myloss.py              # 四个无参考损失函数
├── dataloader.py          # 数据加载器
├── lowlight_train.py      # 训练脚本
├── lowlight_test.py       # 测试脚本
├── camera_enhance.py      # 实时摄像头增强
├── app.py                 # FastAPI 服务
├── export_onnx.py         # ONNX 导出与速度对比
├── gradio_app.py          # Gradio 演示界面
└── requirements.txt       # 依赖列表
```

---
## 鸣谢 (Acknowledgements)

本项目核心算法参考了以下开源仓库：

* **Zero-DCE (CVPR 2020)**: [Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
* **论文引用**: 
    ```text
    @inproceedings{guo2020zero,
      title={Zero-reference deep curve estimation for low-light image enhancement},
      author={Guo, Chunle and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
      booktitle={CVPR},
      year={2020}
    }
    ```

## 参考论文

> Chunle Guo, Chongyi Li, et al. *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*. CVPR 2020.
