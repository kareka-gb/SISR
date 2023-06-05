import torch
import random
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio as PSNR

def plot_evaluation_curves(results):
    """
    Plots training curves using a results dataframe

    Args:
    `results`   - A pandas dataframe containing columns with titles `Epoch`, `Train Loss`, `Validation Loss`, `Train Accuracy`, `Validation Accuracy`
    """
    epoch = results['Epoch']
    tl = results['Train Loss']
    tp = results['Train PSNR']
    vl = results['Validation Loss']
    vp = results['Validation PSNR']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch, tl, label='Train Loss')
    plt.plot(epoch, vl, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.plot(epoch, tp, label='Train PNSR')
    plt.plot(epoch, vp, label='Validation PSNR')
    plt.title('PSNR')
    plt.legend()

def resolve_and_plot_random(model: torch.nn.Module,
                            data: torch.utils.data.Dataset,
                            device: torch.device=torch.device('cpu')):
    lr_image, hr_image = data[random.randint(0, len(data))]
    model.eval()
    with torch.inference_mode():
        sr_image = model(lr_image.unsqueeze(dim=0).to(device)).squeeze().cpu()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(lr_image.permute(1, 2, 0))
    plt.title("LR image")
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plt.imshow(sr_image.permute(1, 2, 0))
    plt.title("SR image")
    plt.axis(False)

    plt.subplot(1, 3, 3)
    plt.imshow(hr_image.permute(1, 2, 0))
    plt.title("HR image")
    plt.axis(False);


def bicubic(img: torch.Tensor,
            scale: int=4):
    h, w = img.shape[1], img.shape[2]
    return transforms.functional.resize(img, size=(h*scale, w*scale), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)


def compare_with_bicubic(model: torch.nn.Module,
                         data: torch.utils.data.Dataset,
                         scale: int=4,
                         device: torch.device=torch.device('cpu')):
    lr_image, hr_image = data[random.randint(0, len(data))]
    psnr = PSNR(data_range=1.0)
    model.eval()
    with torch.inference_mode():
        sr_image = model(lr_image.unsqueeze(dim=0).to(device)).squeeze().cpu()
    
    bicubic_image = bicubic(lr_image, scale=scale)
    
    print("----------------------------------------------------------------")
    print(f"PSNR b/w SR and HR images     : {psnr(sr_image, hr_image)}")
    print(f"PSNR b/w Bicubic and HR images: {psnr(bicubic_image, hr_image)}")
    print("----------------------------------------------------------------")
    
    plt.figure(figsize=(10, 8))
    img_list = list([lr_image, sr_image, hr_image, lr_image, bicubic_image, hr_image])
    titles = list(['LR image', "SR image", "HR image", "LR image", "Bicubic image", "HR image"])
    
    for i, (img, title) in enumerate(zip(img_list, titles)):
        plt.subplot(2, 3, i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(title)
        plt.axis(False)
