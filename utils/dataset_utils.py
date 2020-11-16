
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
from utils.image_utils import load_image
from utils.torch_utils import numpy_to_variable
 
class oriDataset(Dataset):
    def __init__(self, image_dir, resize_height=256, resize_width=256, repeat=1):
        self.image_label_list = []
        for image_count, image_name in enumerate(tqdm(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            self.image_label_list.append(image_name)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()
 
 
    def __getitem__(self, i):
        index = i % self.len
        image_name = self.image_label_list[index]
        label = [image_name]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        img = numpy_to_variable(img)
        return img, label
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
 
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list
   
    def load_data(self, path, resize_height, resize_width, normalization):
        image = load_image((resize_height,resize_width), data_format='channels_first', abs_path=True, fpath=path)
        return image
 
    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data