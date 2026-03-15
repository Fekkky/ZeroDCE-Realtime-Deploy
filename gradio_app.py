import torch
import model
import gradio as gr
from torchvision import transforms
from PIL import Image
import torchvision
import io

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DCE_net = model.enhance_net_nopool().to(device)
DCE_net.load_state_dict(torch.load("snapshots/best_model_best_3.pth", map_location=device))
DCE_net.eval()

transform = transforms.Compose([transforms.ToTensor()])

def enhance(image):
    # 限制最大尺寸
    max_size = 1200
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _, enhanced, _ = DCE_net(input_tensor)

    # 转回PIL Image
    output = enhanced.squeeze(0).cpu().clamp(0, 1)
    result = transforms.ToPILImage()(output)
    return result

# 构建界面
demo = gr.Interface(
    fn=enhance,
    inputs=gr.Image(type="pil", label="上传低光照图片"),
    outputs=gr.Image(type="pil", label="增强结果"),
    title="Zero-DCE 低光照图像增强",
    description="上传一张低光照图片，模型将自动进行亮度增强。",
    examples=["data/test_data/low/1.png"]  # 可选，放一张示例图
)

demo.launch(
    share=False,
    server_name="127.0.0.1",
    server_port=7860,
    show_error=True
)