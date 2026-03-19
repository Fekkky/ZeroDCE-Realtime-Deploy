import torch
import model
import time
import onnxruntime as ort
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DCE_net = model.enhance_net_nopool().to(device)
DCE_net.load_state_dict(torch.load(r"snapshots\best_model_best_3.pth"))
DCE_net.eval()

sess = ort.InferenceSession("snapshots\dce_net.onnx",
        providers=['CUDAExecutionProvider'])

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

input_shape = (1,3,400,600)
x_gpu = torch.randn(input_shape).cuda()
y_gpu = torch.empty(input_shape).cuda()

io_binding = sess.io_binding()
io_binding.bind_input(
    name=input_name,
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=input_shape,
    buffer_ptr = x_gpu.data_ptr()
    
)
io_binding.bind_output(
    name=output_name,
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=input_shape,
    buffer_ptr=y_gpu.data_ptr()
)

for _ in range(10):
    sess.run_with_iobinding(io_binding)
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    sess.run_with_iobinding(io_binding)
    torch.cuda.synchronize() 
onnx_time = (time.time()-start)/100


with torch.no_grad():
    for _ in range(10):
        _ = DCE_net(x_gpu)
    
    torch.cuda.synchronize()
    
    
    start = time.time()
    for _ in range(100):
        _ = DCE_net(x_gpu)
    torch.cuda.synchronize()
    pytorch_time = (time.time()-start)/100
print(f"pytorch平均推理时间：{pytorch_time*1000:.2f}ms")
print(f"ort平均推理时间：{onnx_time*1000:.2f}ms")
print(f"速度提升: {pytorch_time/onnx_time:.2f}x")

