import cv2
import torch
import numpy as np
import model
import time
from torchvision import transforms
from PIL import Image

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DCE_net = model.enhance_net_nopool().to(device)
DCE_net.load_state_dict(torch.load("snapshots/best_model_best_3.pth", map_location=device))
DCE_net.eval()
print(f"模型加载成功，使用设备：{device}")

transform = transforms.Compose([transforms.ToTensor()])

def enhance_frame(frame):
    """对单帧图像进行增强"""
    # OpenCV读取的是BGR，转为RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, enhanced, _ = DCE_net(input_tensor)

    # 转回numpy BGR格式
    enhanced_np = enhanced.squeeze(0).cpu().clamp(0, 1).numpy()
    enhanced_np = (enhanced_np * 255).astype(np.uint8)
    enhanced_np = np.transpose(enhanced_np, (1, 2, 0))  # CHW -> HWC
    enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
    enhanced_bgr = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)
    return enhanced_bgr

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头已启动，按Q退出")

fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    start = time.time()
    enhanced = enhance_frame(frame)
    fps = 1.0 / (time.time() - start)
    fps_list.append(fps)

    # 左右拼接对比显示
    combined = np.hstack([frame, enhanced])

    # 显示FPS
    cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Original", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Enhanced", (frame.shape[1] + 10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Zero-DCE Real-time Enhancement", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

avg_fps = np.mean(fps_list)
print(f"平均FPS：{avg_fps:.1f}")
print(f"这个数字可以写入简历：实时处理速度达到 {avg_fps:.0f} FPS")