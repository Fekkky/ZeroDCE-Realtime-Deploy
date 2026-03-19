import torch
import torchvision
import os
from PIL import Image
import model
from torchvision import transforms

def lowlight(image_path,model_path):
    print(f"\nProcessing image: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"目标文件未找到")
    
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        data_lowlight_pil = Image.open(image_path).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((256,256),Image.Resampling.LANCZOS),
            transforms.ToTensor()
        ])
        data_lowlight = transform(data_lowlight_pil)
        
        data_lowlight = data_lowlight.to(device).unsqueeze(0)
        
        DCE_NET = model.enhance_net_nopool().to(device)
        DCE_NET.load_state_dict(torch.load(model_path,map_location = device))
        DCE_NET.eval()
        
        with torch.no_grad():
            _,enhance_image,_ = DCE_NET(data_lowlight)
        
        result_path = image_path.replace('try_data',"result")
        result_path = result_path.replace('3',"3_1")
        result_dir = os.path.dirname(result_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            print(f"Created directory:{result_dir}")
        
        torchvision.utils.save_image(enhance_image,result_path)
        print(f"Enhanced image saved to:{result_path}")
        
    except Exception as e:
        raise e
    
if __name__ == '__main__':
    image_path = r"data\try_data\3.jpg"
    model_path = r"snapshots\best_model_best_3.pth"
    lowlight(image_path,model_path)
        