import torchvision.transforms.functional as TF

from random import randint
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Any


class Div2k(Dataset):
    def __init__(self, hr_path: str, lr_path: str, subset: str = 'train', scale: int = 4, patch_size: Tuple[int, int] = (24, 24)):
        super().__init__()
        hr_path = Path(hr_path)
        lr_path = Path(lr_path)

        self.scale = scale
        self.subset = subset
        self.patch_size = patch_size
        self.hr_img_paths = sorted(list(hr_path.glob('*.png')))
        self.lr_img_paths = sorted(list(lr_path.glob('*.png')))

    def __len__(self):
        return len(self.hr_img_paths)
    
    def __getitem__(self, index) -> Any:
        lr, hr = self._load_images(index)
        return TF.to_tensor(lr), TF.to_tensor(hr)
    
    def _load_images(self, index) -> Tuple[Image.Image, Image.Image]:
        hr_img = Image.open(self.hr_img_paths[index])
        lr_img = Image.open(self.lr_img_paths[index])

        # if self.subset == 'train':
        lx, ly = self._get_patch(lr_img.size)
        hx = lx * self.scale
        hy = ly * self.scale
        return self._random_transform(lr_img.crop((lx, ly, lx+self.patch_size[0], ly+self.patch_size[1])), hr_img.crop((hx, hy, hx+self.scale*self.patch_size[0], hy+self.scale*self.patch_size[1])))
        # elif self.subset == 'valid':
        #     return lr_img, hr_img
        # else:
        #     NotImplementedError()
    
    def _get_patch(self, lr_img_size: Tuple[int, int]) -> Tuple[int, int]:
        assert self.patch_size[0] < lr_img_size[0] and self.patch_size[1] < lr_img_size[1], "Patch size should be lower than the LR image size"
        stx = randint(0, lr_img_size[0] - self.patch_size[0])
        sty = randint(0, lr_img_size[1] - self.patch_size[1])
        return stx, sty
    
    def _random_transform(self, imga: Image.Image, imgb: Image.Image) -> Tuple[Image.Image, Image.Image]:
        imga, imgb = self._random_rotate(imga, imgb)
        imga, imgb = self._random_flip_vertical(imga, imgb)
        return self._random_flip_horizontal(imga, imgb)
    
    def _random_flip_vertical(self, imga: Image.Image, imgb: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if randint(0, 2) == 1:
            return imga.transpose(Image.FLIP_LEFT_RIGHT), imgb.transpose(Image.FLIP_LEFT_RIGHT)
        return imga, imgb
    
    def _random_flip_horizontal(self, imga: Image.Image, imgb: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if randint(0, 2) == 1:
            return imga.transpose(Image.FLIP_TOP_BOTTOM), imgb.transpose(Image.FLIP_TOP_BOTTOM)
        return imga, imgb
    
    def _random_rotate(self, imga: Image.Image, imgb: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if randint(0, 2) == 1:
            return imga.transpose(Image.ROTATE_90), imgb.transpose(Image.ROTATE_90)
        return imga, imgb


