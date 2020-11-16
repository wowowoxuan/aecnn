import sys
from models.aegandc import aegenerator
from models.vgg import Vgg16
import os
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.utils as vutils
from utils.dataset_utils import oriDataset
from utils.image_utils import load_image
from utils.torch_utils import numpy_to_variable,variable_to_numpy
from PIL import Image
from tqdm import tqdm
import datetime
import numpy as np
generator = aegenerator().cuda()
dataset = oriDataset(image_dir='',resize_height=224,resize_width=224)#type clean image path here
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,shuffle=False)
generator.load_state_dict(torch.load('')) #weight path
generator.eval()
i = 0
start = datetime.datetime.now()
for data, filename in dataloader:
    i += 1
    print(i)
    advs = generator(data)

    advs = torch.max(torch.min(advs, data + 16/255), data - 16/255)
    advs = torch.clamp(advs, 0, 1)
    advs_np = variable_to_numpy(advs)
    for idx, adv_np in enumerate(advs_np):
        image_pil = Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0)))
        file = filename[0][idx].replace('.jpg','.png')
        folderpath = '/'#output folder
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        path = folderpath + file
        image_pil.save(path)
    #print(filename[0][0])
end = datetime.datetime.now()
print(end-start)
