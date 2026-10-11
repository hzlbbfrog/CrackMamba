"""Dataset loaders for DeepCrack and Steelcrack segmentation."""

import torch.utils.data as data
from torchvision.transforms import transforms
import PIL.Image as Image
import os
import random
import numpy as np
import cv2
from glob import glob
import torch
from pathlib import Path


class DeepcrackDatasetrandom(data.Dataset):
    """Load the DeepCrack 1800 splits with resized and cropped training views."""
    def __init__(self, Current_dir, state):
        self.state = state
        self.aug = True

        # DeepCrack split directories.
        self.train_root = Current_dir + '/Dataset/DeepCrack/DeepCrack_1800' + '/Train_Aug'
        self.val_root = Current_dir + '/Dataset/DeepCrack/DeepCrack_1800' + '/Validation'
        self.test_root = Current_dir + '/Dataset/DeepCrack/DeepCrack_1800' + '/Test'
        self.resize = 480
        self.cropsize = 384
        self.pics, self.masks = self.getDataPath()

        if self.state == 'Train':
            self.transform1, self.transform2, self.target_transform1, self.target_transform2 = self.get_transform()
        else:
            self.transform, self.target_transform = self.get_transform()

    def getDataPath(self):
        assert self.state =='Train' or self.state == 'Validation' or self.state =='Test'
        if self.state == 'Train':
            root = self.train_root
        if self.state == 'Validation':
            root = self.val_root
        if self.state == 'Test':
            root = self.test_root

        pics = glob(root + '/images/*')
        masks = glob(root + '/masks/*')
        return pics,masks

    def get_transform(self):

        if self.state == 'Train':
            transform1 = transforms.Compose([
                    transforms.ToTensor(),  # Scale RGB values to [0, 1].
                    transforms.Resize((self.resize, self.resize)),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # Normalize RGB values to [-1, 1].

            transform2 = transforms.Compose([
                    transforms.ToTensor(),  # Scale RGB values to [0, 1].
                    transforms.RandomCrop((self.cropsize, self.cropsize)),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # Normalize RGB values to [-1, 1].

            target_transform1 = transforms.Compose([
                    transforms.ToTensor(),  # Scale mask values to [0, 1].
                    transforms.Resize((self.resize, self.resize))])

            target_transform2 = transforms.Compose([
                    transforms.ToTensor(),  # Scale mask values to [0, 1].
                    transforms.RandomCrop((self.cropsize, self.cropsize))
                    ])

            return transform1, transform2, target_transform1, target_transform2

        else:
            transform1 = transforms.Compose([
                    transforms.ToTensor(),  # Scale RGB values to [0, 1].
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # Normalize RGB values to [-1, 1].

            target_transform1 = transforms.Compose([
                    transforms.ToTensor()])  # Scale mask values to [0, 1].

            return transform1, target_transform1


    def __getitem__(self, index):
        """Return images, masks, and source paths for one sample.

        Training returns two image-mask pairs followed by their source paths.
        Validation and testing return one image-mask pair and the source paths.
        """
        pic_path = self.pics[index]
        mask_path = self.masks[index]

        pic = self.rgb_loader(pic_path)
        mask = self.binary_loader(mask_path)

        if self.state == 'Train':
            img_resize = self.transform1(pic)
            label_resize = self.target_transform1(mask)

            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img_crop = self.transform2(pic)
            torch.random.manual_seed(seed)
            label_crop = self.target_transform2(mask)

            return img_resize, label_resize, img_crop, label_crop, pic_path, mask_path

        else:
            img = self.transform(pic)
            label = self.target_transform(mask)

            return img, label, pic_path, mask_path

    def __len__(self):
        return len(self.pics)

    def rgb_loader(self, path):
        img = Image.open(path)
        return img.convert('RGB')

    def binary_loader(self, path):
        img = Image.open(path)
        return img.convert('L')


class Get_Steel_Cracks_with_Edge(data.Dataset):
    """Load Steelcrack images, segmentation masks, and edge masks."""
    def __init__(self, Current_dir, state):
        self.state = state

        data_path = Current_dir + '/Dataset/' + 'Steelcrack/' + state
        image_root = data_path + '/images/'
        gt_root = data_path + '/masks/'
        gt_edge_root = data_path + '/edges/'

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.gt_edges = [gt_edge_root + f for f in os.listdir(gt_edge_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gt_edges = sorted(self.gt_edges)

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),  # Scale RGB values to [0, 1].
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # Normalize RGB values to [-1, 1].

        self.gt_transform = transforms.Compose([transforms.ToTensor()])  # Scale mask values to [0, 1].

    def __getitem__(self, index):
        """Return a normalized image, segmentation mask, and sample name.

        Training also returns the edge mask before the sample name.
        """
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        gt_edge = self.binary_loader(self.gt_edges[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        gt_edge = self.gt_transform(gt_edge)

        name = self.images[index].split('/')[-1][0:-4]
        if self.state == 'Train':
            return image, gt, gt_edge, name
        else:
            return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
