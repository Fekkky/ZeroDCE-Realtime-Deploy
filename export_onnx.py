import torch
import model
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
DCE_net = model.enhance_net_nopool().to(device)
DCE_net.load_state_dict(torch.load("snapshots/best_model_best_3.pth", map_location=device))
DCE_net.eval()

# 构造dummy输入（和实际推理尺寸一致）
dummy_input = torch.randn(1, 3, 400, 600).to(device)

# 导出ONNX
torch.onnx.export(
    DCE_net,
    dummy_input,
    "snapshots/dce_net.onnx",
    input_names=["input"],
    output_names=["enhanced_image_1", "enhanced_image", "A"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "enhanced_image": {0: "batch", 2: "height", 3: "width"}
    },
    opset_version=11
)
print("ONNX导出成功")

# 对比推理速度
dummy_input_cpu = torch.randn(1, 3, 400, 600)

# PyTorch推理速度
with torch.no_grad():
    for _ in range(10):
        DCE_net(dummy_input)
        
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = DCE_net(dummy_input)
    pytorch_time = (time.time() - start) / 100
print(f"PyTorch平均推理时间: {pytorch_time*1000:.2f} ms")

# ONNX Runtime推理速度
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("snapshots/dce_net.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_np = dummy_input_cpu.numpy()

for _ in range(10):
    sess.run(None, {"input": input_np})
    
    
start = time.time()
for _ in range(100):
    sess.run(None, {"input": input_np})
onnx_time = (time.time() - start) / 100
print("实际使用的Provider:", sess.get_providers())
print(f"ONNX Runtime平均推理时间: {onnx_time*1000:.2f} ms")
print(f"速度提升: {pytorch_time/onnx_time:.2f}x")