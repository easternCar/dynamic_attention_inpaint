import sys
import torch
import torch.utils.data as data
from os import listdir
from utils.tools import is_image_file, normalize
import os
import cv2
import random

import torchvision.transforms as transforms

def pil_loader(path, chan='RGB'):
    try:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert(chan)

    except IOError:
        print('Cannot load image ' + path)



def img_loader(path, IMSIZE=128):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)

            w, h = img.shape[0], img.shape[1]
            if w < IMSIZE or h < IMSIZE or w > IMSIZE or IMSIZE:
                img = cv2.resize(img, (IMSIZE, IMSIZE), interpolation=cv2.INTER_CUBIC)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)

            return img
    except IOError:
        print('Cannot load image ' + path)


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, transform, with_subfolder=False, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.transform = transform
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        self.default_loader = img_loader
        print(str(len(self.samples)) + "  items found")

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = self.default_loader(path)

        #w, h = img.shape[0], img.shape[1]
        #if w < self.image_shape[0] or h < self.image_shape[1] or w > self.image_shape[0] or h > self.image_shape[1]:
        #    img = cv2.resize(img, (self.image_shape[0], self.image_shape[1]), interpolation=cv2.INTER_CUBIC)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        '''
        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)
        '''

        if self.return_name:
            return self.samples[index], img
        else:
            return img

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)
