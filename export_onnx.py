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
    opset_version=14
)
print("ONNX导出成功")

