import sys
from models.ctcgan import aegenerator

from models.vgg import Vgg16
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.utils as vutils
from utils.dataset_utils import oriDataset
from utils.torch_utils import variable_to_numpy
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from utils import ssim
from vgg import Vggcon
from models.gabor import GaborConv2d
import torchvision

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
best_loss = 100
batchsize = 12
generator = aegenerator().cuda()
vggmodel = models.vgg16(pretrained=True).cuda().eval()
optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) # initial is2. 0.0002
print(generator.modules())
loss_func = nn.MSELoss()
CEloss = nn.CrossEntropyLoss()
dataset = oriDataset(image_dir='',resize_height=224,resize_width=224)#train set
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=True)

num_iters = 20
gabor = GaborConv2d().cuda().eval()
lr = 0.0002
#vgg = Vggcon().cuda().eval()
for epoch in range(num_iters):
    i = 0
    print('epoch:' + str(epoch) + 'begin')
    test_celoss = 0
    test_ssimloss = 0
    test_mseloss = 0
    test_fl = 0
    test_loss = 0
    test_count = 0
    for data, filename in dataloader:
        test_count += 1
        #print(data.shape)
        generator.zero_grad()
        output = generator(data)
        #print(output.shape)
        output = torch.max(torch.min(output, data + 16/255), data - 16/255)
        output = torch.clamp(output, 0, 1)
        gabor1 = gabor(output)
        gabor2 = gabor(data)
        gaborthreshold = gabor2<=0
        gaborthreshold = gaborthreshold.float()

        mseloss = loss_func(gabor1*gaborthreshold,gabor2*gaborthreshold)
        #mseloss = loss_func(gabor1,gabor2)
        #loss3 = 1 - ssim.ssim(data,output)
        #mseloss = torch.std(output-data)
        
        if i == 0:
            advs_np = variable_to_numpy(output)
            image_pil = Image.fromarray(np.transpose((advs_np[0] * 255).astype(np.uint8), (1, 2, 0)))
            file = filename[0][0].replace('.jpg','.png')
            path = './testimages/' + str(epoch) + '.png'
            image_pil.save(path)
        t_mean = torch.FloatTensor(mean).view(1,3,1,1).expand(1,3, 224, 224).cuda()
        t_std = torch.FloatTensor(std).view(1,3,1,1).expand(1,3, 224, 224).cuda()
        output = (output-t_mean)/t_std
        data = (data-t_mean)/t_std
        vggout1 = vggmodel(output)
        vggout2 = vggmodel(data)
        #print(vggout2.shape)
        # y_c_features = vgg(output)
        # y_hat_features = vgg(data)
        # f_l = loss_func(y_c_features[1],y_hat_features[1])
        label = vggout2.argmax(dim=1)
        #vggout2 = F.softmax(vggout2, dim=-1)
        #label = vggout2.argmax(dim=1)
        #print(label)
        celoss = CEloss(vggout1,label)
        celoss = 1/celoss
        #celoss = 1/loss_func(vggout1, vggout2)
        #print(celoss)
        print(mseloss)
        #print(loss3)
        #print(f_l)
        loss = celoss + 80 * mseloss #+ 3*mseloss#+ mseloss #+ 0.01 * f_l+ loss3 
        #print(loss3)
        print(celoss)
        print(loss)
        loss.backward()
        optimizer.step()
        print('batch:' + str(i) + 'end')
        test_celoss += celoss
        #test_ssimloss += loss3
        test_mseloss += mseloss
       # test_fl += f_l
        test_loss += loss
        i += 1
        #break
        

        
    average_celoss = test_celoss/test_count
    #average_ssimloss = test_ssimloss/test_count
    average_mseloss = test_mseloss/test_count
    #average_fl = test_fl/test_count
    average_loss = test_loss/test_count
    a = open('./lossfile/multice01ssim.txt','a')
    #loss3numpy = average_ssimloss.cpu().detach().numpy()
    ce1numpy = average_celoss.cpu().detach().numpy()
    lossnumpy = average_loss.cpu().detach().numpy()
    msenumpy = average_mseloss.cpu().detach().numpy()
    #fl_numpy = average_fl.cpu().detach().numpy()
    #print(loss)
    a.write('epoch: ' + str(epoch) + 'celoss'+' ' + str(ce1numpy) + '\n')
    #a.write('epoch: ' + str(epoch) + 'ssim'+' ' + str(loss3numpy) + '\n')
    a.write('epoch: ' + str(epoch) + 'gaborloss'+' ' + str(msenumpy) + '\n')
    a.write('epoch: ' + str(epoch) + ' ' + str(lossnumpy) + '\n')
    torch.save(generator.state_dict(), '/data1/wchai01/ctcweight/vgg/celgabor/aegenerator_epoch_%d.pth' % (epoch))#weight dc is the celoss only
    print('=============================================================================\n')
    print('epoch:' + str(epoch) + 'end')
    #break
    
  
    


