import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch


def load_model(model_path, device):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model


def super_resolution(img, model, device):
    # 将图像归一化并转换为Tensor
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # 使用模型进行超分辨率处理
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # 将输出转换回OpenCV格式
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    return output


# 初始化模型
model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = load_model(model_path, device)

# 测试图像文件夹
test_img_folder = 'LR/*'

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)

    # 读取图像
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # 进行超分辨率处理
    output = super_resolution(img, model, device)

    # 保存结果
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)