from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from utils.transform import ReLabel, ToLabel, Scale, HorizontalFlip, VerticalFlip, ColorJitter
from utils.transform import ReLabelBinary
import random

class Dataset_Aug(torch.utils.data.Dataset):

    def __init__(self, img_dir, mask_dir):
        self.size = (128,128)
#        self.root = root
        if not os.path.exists(img_dir):
            raise Exception("[!] {} not exists.".format(img_dir))
        if not os.path.exists(mask_dir):
            raise Exception("[!] {} not exists.".format(mask_dir))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabelBinary(127),
        ])
        #sort file names
        self.input_paths = sorted(glob('{}/*.jpg'.format(img_dir)))
        self.label_paths = sorted(glob('{}/*.jpg'.format(mask_dir)))
        #self.name = os.path.basename(root)
        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}")

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index]).convert('RGB')
        # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        label = Image.open(self.label_paths[index]).convert('P')

        image = self.img_resize(image)
        # image_hsv = self.img_resize(image_hsv)
        label = self.label_resize(label)
        # brightness_factor = 1 + random.uniform(-0.4,0.4)
        # contrast_factor = 1 + random.uniform(-0.4,0.4)
        # saturation_factor = 1 + random.uniform(-0.4,0.4)
        # hue_factor = random.uniform(-0.1,0.1)
        # gamma = 1 + random.uniform(-0.1,0.1)

        #randomly flip images
        if random.random() > 0.5:
            image = HorizontalFlip()(image)
            # image_hsv = HorizontalFlip()(image_hsv)
            label = HorizontalFlip()(label)
        if random.random() > 0.5:
            image = VerticalFlip()(image)
            # image_hsv = VerticalFlip()(image_hsv)
            label = VerticalFlip()(label)

        #randomly crop image to size 128*128
        #w, h = image.size
        #th, tw = (128,128)
        #x1 = random.randint(0, w - tw)
        #y1 = random.randint(0, h - th)
        #if w == tw and h == th:
        #    image = image
        #    # image_hsv = image_hsv
        #    label = label
        #else:
        #    if random.random() > 0.5:
        #        image = image.resize((128,128),Image.BILINEAR)
        #        # image_hsv = image_hsv.resize((128,128),Image.BILINEAR)
        #        label = label.resize((128,128),Image.NEAREST)
        #    else:
        #        image = image.crop((x1, y1, x1 + tw, y1 + th))
        #        # image_hsv = image_hsv.crop((x1, y1, x1 + tw, y1 + th))
        #        label = label.crop((x1, y1, x1 + tw, y1 + th))


        # angle = random.randint(-20, 20)        
        rand_number = random.random()
        if 0.25 < rand_number < 0.5:
             image = image.rotate(90, resample=Image.BILINEAR)
             label = label.rotate(90, resample=Image.BILINEAR)
        elif 0.5 < rand_number < 0.75:
             image = image.rotate(180, resample=Image.BILINEAR)
             label = label.rotate(180, resample=Image.BILINEAR)
        elif 0.75 < rand_number < 1.0:
             image = image.rotate(270, resample=Image.BILINEAR)
             label = label.rotate(270, resample=Image.BILINEAR)
        # image_hsv = image_hsv.rotate(angle, resample=Image.BILINEAR)
        # label = label.rotate(angle, resample=Image.NEAREST)
        image = self.img_transform(image)
        # image_hsv = self.hsv_transform(image_hsv)
        # image = torch.cat([image,image_hsv],0)
        label = self.label_transform(label)

        return {'image':image, 'mask':label}

    def __len__(self):
        return len(self.input_paths)

class Dataset_No_Aug(torch.utils.data.Dataset):

    def __init__(self, img_dir, mask_dir):
        self.size = (128,128)
#        self.root = root
        if not os.path.exists(img_dir):
            raise Exception("[!] {} not exists.".format(img_dir))
        if not os.path.exists(mask_dir):
            raise Exception("[!] {} not exists.".format(mask_dir))
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabelBinary(127),
        ])
        #sort file names
        self.input_paths = sorted(glob('{}/*.jpg'.format(img_dir)))
        self.label_paths = sorted(glob('{}/*.jpg'.format(mask_dir)))
        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}")


    def __getitem__(self, index):
        image = Image.open(self.input_paths[index]).convert('RGB')
        label = Image.open(self.label_paths[index]).convert('P')

        image = self.img_resize(image)
        label = self.label_resize(label)

        image = self.img_transform(image)
        label = self.label_transform(label)

        return {'image':image, 'mask':label}

    def __len__(self):
        return len(self.input_paths)


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, root):
        size = (128,128)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

        ])
        self.hsv_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Scale(size, Image.NEAREST),
            ToLabel(),
            ReLabel(255, 1),
        ])
        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.jpg'.format("ISIC-2017_Test_v2_Data"))))
        self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.png'.format("ISIC-2017_Test_v2_Part1_GroundTruth"))))
        #self.name = os.path.basename(root)
        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index]).convert('RGB')
        # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        label = Image.open(self.label_paths[index]).convert('P')

        if self.img_transform is not None:
            image = self.img_transform(image)
            # image_hsv = self.hsv_transform(image_hsv)
        else:
            image = image
            # image_hsv = image_hsv

        if self.label_transform is not None:
            label = self.label_transform(label)
        else:
            label = label
        # image = torch.cat([image,image_hsv],0)

        return image, label

    def __len__(self):
        return len(self.input_paths)




class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}



