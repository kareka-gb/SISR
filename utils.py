import torch
import random
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms

# function to calculate psnr
def psnr(img1: torch.Tensor, img2: torch.Tensor, max_value = 1.0):
    """
    Returns psnr of a single image
    """
    if torch.equal(img1, img2):
        return torch.tensor(float('inf'))
    mse = torch.nn.functional.mse_loss(img1, img2)
    return 10 * torch.log10((max_value ** 2) / mse)

def batch_psnr(preds: torch.Tensor, target: torch.Tensor, reduction='mean'):
    """
    Args:
    preds - predicted image batch
    target - target image batch
    reduction - which type of reduction to use
    
    Returns average psnr over a batch. Expects a batch of images to be of shape [batch_size, num_channels, height, width]
    """

    assert preds.shape == target.shape, "input and target are not of the same shape"

    n = preds.shape[0]
    psnrs = torch.zeros(n).type(torch.float)
    for i in range(n):
        psnrs[i] = psnr(preds[i], target[i])
    
    if reduction == 'mean':
        return psnrs.mean()
    elif reduction == 'sum':
        return psnrs.sum()
    else:
        NotImplementedError

def plot_img(img, title="image"):
    plt.imshow(img)
    plt.title(title)
    plt.axis(False)

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
    plot_img(lr_image.permute(1, 2, 0), "LR image")

    plt.subplot(1, 3, 2)
    plot_img(sr_image.permute(1, 2, 0), "SR image")

    plt.subplot(1, 3, 3)
    plot_img(hr_image.permute(1, 2, 0), "HR image")


def bicubic(img: torch.Tensor,
            scale: int=4):
    h, w = img.shape[1], img.shape[2]
    return transforms.functional.resize(img, size=(h*scale, w*scale), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)


def compare_with_bicubic(model: torch.nn.Module,
                         data: torch.utils.data.Dataset,
                         scale: int=4,
                         device: torch.device=torch.device('cpu')):
    lr_image, hr_image = data[random.randint(0, len(data))]
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
