from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import os


MY_PATH = os.path.dirname(__file__)


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    def __call__(self, img):
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return img


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img


def transform(sample):
    composed_transforms = transforms.Compose([
        FixScaleCrop(crop_size=224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])
    return composed_transforms(sample)


def make_shuffle_path(image_folder):
    test_A, test_B = [], []
    imageList = os.listdir(image_folder)
    for i in range(len(imageList)):
        for j in range(i + 1, len(imageList)):
            test_A.append(imageList[i])
            test_B.append(imageList[j])
    return test_A, test_B


class MyDataset(data.Dataset):

    def __init__(self, test_data_folder):
        self.pathA, self.pathB = make_shuffle_path(test_data_folder)

    def __getitem__(self, index):
        imageA = Image.open(os.path.join(MY_PATH, 'data/61/') + self.pathA[index]).convert(
            'RGB')
        imageB = Image.open(os.path.join(MY_PATH, 'data/61/') + self.pathB[index]).convert(
            'RGB')

        imageA = transform(imageA)
        # imageA = imageA.unsqueeze(0)
        imageB = transform(imageB)
        # imageB = imageB.unsqueeze(0)
        return imageA, imageB, self.pathA[index], self.pathB[index]

    def __len__(self):
        return len(self.pathA)


def make_loader(test_data_folder):
    test_data = MyDataset(test_data_folder)
    testloader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return testloader
