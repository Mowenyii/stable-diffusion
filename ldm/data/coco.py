import os


import torch
import torch.utils.data as data
from PIL import Image
# from pycocotools.coco import COCO #We're using the MS COCO dataset
from torchvision import datasets, transforms
from collections import defaultdict
import json

class COCO:
    def __init__(self, annotation_file=None, split="test"):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.imgs = dict(),dict()
        self.split=split
        if not annotation_file == None:

            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))

            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        imgs = {}


        if 'images' in self.dataset:
            for img in self.dataset['images']:
                # if img['split']==self.split:
                imgs[img['cocoid']] = img

        # create class members
        self.imgs = imgs

# dataset["images"][0].keys()
# dict_keys(['filepath', 'sentids', 'filename', 'imgid', 'split', 'sentences', 'cocoid'])
# with open(annotation_file, 'r') as f:
#     dataset = json.load(f)

class DataLoader(data.Dataset):
    def __init__(self, root, json,  transform=None):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs)

        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco

        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)


        caption = []

        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


root="/home/heng_zhang/data/root/val2014"
annotation_file="/home/heng_zhang/data/root/karpathy/dataset_coco.json"
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 好像是yaml要求256
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])
coco = DataLoader(root=root, json=annotation_file, transform=transform)

# Dataloader for MS COCO dataset
# 1. This will return (images, captions, lengths) for every iteration.
# 2. images: tensor of shape (batch_size, 3, 224, 224).
# 3. captions: tensor of shape (batch_size, padded_length).
# 4. lengths: list indicating valid length for each caption. length is (batch_size).
data_loader = torch.utils.data.DataLoader(dataset=coco,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=2)
