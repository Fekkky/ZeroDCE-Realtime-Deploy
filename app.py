import io
import torch
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from contextlib import asynccontextmanager

# --- 配置与初始化 ---
MODEL_PATH = "snapshots/dce_net.onnx"
INPUT_SHAPE = (1, 3, 640, 480)  

class ORTService:
    def __init__(self, model_path):
        # 开启所有优化并使用 CUDA
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=opts, 
            providers=[('CUDAExecutionProvider', {'enable_cuda_graph': '1'})]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        

        self.in_gpu = torch.empty(INPUT_SHAPE, dtype=torch.float32, device='cuda')
        self.out_gpu = torch.empty(INPUT_SHAPE, dtype=torch.float32, device='cuda')
        
        # 绑定 IO
        self.io_binding = self.session.io_binding()
        self.io_binding.bind_input(
            name=self.input_name, device_type='cuda', device_id=0,
            element_type=np.float32, shape=INPUT_SHAPE, buffer_ptr=self.in_gpu.data_ptr()
        )
        self.io_binding.bind_output(
            name=self.output_name, device_type='cuda', device_id=0,
            element_type=np.float32, shape=INPUT_SHAPE, buffer_ptr=self.out_gpu.data_ptr()
        )

    def predict(self, img_pil):
        # 1. 预处理：调整大小到固定尺寸并转为 Tensor
        # 注意：如果尺寸变化频繁，则不能使用固定的 io_binding
        img_resized = img_pil.resize((INPUT_SHAPE[3],INPUT_SHAPE[2]),Image.Resampling.LANCZOS)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        
        # 2. 数据拷贝到预分配的 GPU 缓冲区
        temp_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()
        self.in_gpu.copy_(temp_tensor)
        
        # 3. 推理 (Zero-Copy)
        self.session.run_with_iobinding(self.io_binding)
        
        # 4. 后处理
        output = self.out_gpu.squeeze(0).cpu().clamp(0, 1).numpy()
        output = (output.transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(output)

# 使用 lifespan 管理模型生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ort_model = ORTService(MODEL_PATH)
    yield
    del app.state.ort_model

app = FastAPI(lifespan=lifespan)

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 对于单卡环境，直接调用即可
    enhanced_img = app.state.ort_model.predict(img)
    
    # 转换为流输出
    buf = io.BytesIO()
    enhanced_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/jpeg")

@app.get("/health")
async def health():
    return {"status": "running", "engine": "onnxruntime-cuda"}