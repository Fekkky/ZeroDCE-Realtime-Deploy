import io
import torch
import model
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from torchvision import transforms
from PIL import Image
import torchvision

app = FastAPI()

# 启动时加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DCE_net = model.enhance_net_nopool().to(device)
DCE_net.load_state_dict(torch.load("snapshots/best_model_best_3.pth", map_location=device))
DCE_net.eval()

transform = transforms.Compose([transforms.ToTensor()])

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 限制最大尺寸，保持宽高比
    max_size = 1200
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # 推理
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, enhanced, _ = DCE_net(input_tensor)
    
    output = io.BytesIO()
    torchvision.utils.save_image(enhanced, output, format="PNG")
    output.seek(0)
    
    return StreamingResponse(output, media_type="image/png")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}