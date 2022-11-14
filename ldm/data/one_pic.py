import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
import open_clip

#https://github.com/lugiavn/notes/blob/master/fashioniq_tirg.md
class OnePic(Dataset):
    def __init__(self, path,interpolation="bicubic",flip_p=0.,size=None, transform=None):
        super(OnePic, self).__init__()
        self.path=path
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
    def __len__(self):
        return 1#len(self.queries)#len(self.imgs)


    def __getitem__(self, idx):#, raw_img=False):
        example = {}
        with open(self.path+"dev-1000-1-img0.png", 'rb') as f:
            img_s = PIL.Image.open(f)
            img_s = img_s.convert('RGB')

        # default to score-sde preprocessing
        img_s = np.array(img_s).astype(np.uint8)
        crop = min(img_s.shape[0], img_s.shape[1])
        h, w, = img_s.shape[0], img_s.shape[1]
        img_s = img_s[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image_s = Image.fromarray(img_s)
        if self.size is not None:
            image_s = image_s.resize((self.size, self.size), resample=self.interpolation)
        image_s = self.flip(image_s)
        image_s = np.array(image_s).astype(np.uint8)
        example["image"] = (image_s / 127.5 - 1.0).astype(np.float32)
        example["caption"] = "Hang a bright green canopy over the bed."
        return example


class OnePicTrain(OnePic):
    def __init__(self, **kwargs):
        super().__init__(path="/home/wenyi_mo/stable-diffusion/one_pic/",  **kwargs) #


class OnePicValidation(OnePic):
    def __init__(self, **kwargs):
        super().__init__(path="/home/wenyi_mo/stable-diffusion/one_pic/", **kwargs)