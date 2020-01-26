import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from utils import * #is another separate file
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 2, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv7 = nn.Conv2d(1024, 2, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")

seed = 1
torch.cuda.manual_seed(seed)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def weights_init(m):
    if isinstance(m, nn.modules.conv._ConvNd): #nn.Conv3d
        init.xavier_uniform_(m.weight.data, gain = np.sqrt(2.0))
        m.bias.data.fill_(0)
        # torch.nn.init.xavier_uniform_(m.bias.data)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        # init.xavier_uniform_(m.weight.data)
        y = 1/np.sqrt(m.in_features)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0) #0.01

class ConvBlock1D(nn.Module):
    def __init__(self,in_channel, out_channel, activation = F.relu):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.activation = activation

    def forward(self,input):
        output = self.activation(self.bn1(self.conv1(input)))
        return output

class ConvBlock(nn.Module): #here nn.Module is the superclass, UNetConvBlock is the subclass
    def __init__(self,in_channel, out_channel, activation=F.relu): #activation = F.relu
        super(ConvBlock,self).__init__() #UNetConvBlock inherits the functionality of nn.Module
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # self.activation = F.tanh(self.out) #does it require input?
        self.activation = activation

        # init.xavier_uniform_(self.conv1.weight, gain=5/3)
        # init.constant_(self.conv1.bias,0)
        # init.xavier_uniform_(self.conv2.weight,gain=5/3)
        # init.constant_(self.conv2.bias,0)

    def forward(self, input):
        # output = torch.tanh(self.bn1(self.conv1(input)))
        output = self.activation(self.bn1(self.conv1(input)))
        # print('Here')
        # output =torch.tanh(self.bn2(self.conv2(output)))
        output = self.activation(self.bn2(self.conv2(output)))
        return output

class residualUnit(nn.Module):
    def __init__(self,in_size, out_size,activation= F.relu): #in_size, out_size spatial ??
        super(residualUnit,self).__init__()
        self.conv1 = nn.Conv2d(in_size,out_size,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size,out_size, kernel_size=3, stride=1, padding=1) #after this line both channel size will be same which will execute bridge
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size,out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)

        # init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2.0))
        # init.constant_(self.conv1.bias,0)
        # init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        # init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        if self.in_size != self.out_size:
            bridge = self.activation(self.bnX(self.convX(x))) #bridge is basically the convoluted input with same channels as output
            #bridge is channel to channel addition... according to the ResNet paper.
        out = torch.add(out, bridge) #equivalent to multiplied by 2
        return out

# conv_block64_128 = residualUnit(32, 64)
# print('here')

class ResNet_PC(nn.Module):
    def __init__(self, in_channel=3,n_classes=2): #prob % values will be zeroed
        super(ResNet_PC,self).__init__()
        #self.Dropout = nn.Dropout3d(p=prob)
        # self.pool1 = nn.MaxPool3d(2)
        # self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.strConv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        # self.pool2 = nn.MaxPool3d(2)
        # self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        # self.pool3 = nn.MaxPool3d(2) #avgpool with 2 as input reduces the dimension too much
        self.strConv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        # self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.strConv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_block1_32 = ConvBlock(in_channel, 32)
        self.conv_block32_64 = residualUnit(32,64)
        self.conv_block64_128 = residualUnit(64,128)
        self.conv_block128_256 = residualUnit(128,256)
        # self.conv_block256_512 = residualUnit(256,512)
        # self.conv_block512_1024 = residualUnit(512,1024)
        # self.conv1d = ConvBlock1D(in_channel=1, out_channel=2)

        # self.fc1 = nn.Linear(256*11*14*10, n_classes) #512 todo: get the hidden layer size
        self.fc1 = nn.Linear(650752, n_classes) # 2 for out_channel of 1D and n_classes = 256 for real part
        self.fc2 = nn.Linear(n_classes, n_classes) #todo: tanh(), syntax gonna work?

    def forward(self, x): #torch.Size([2, 1, 86, 110, 78])
        # print(x.shape) #torch.Size([4, 3, 326, 490])
        out = self.conv_block1_32(x) #torch.Size([2, 32, 86, 110, 78])
        # print(out.shape) #torch.Size([4, 32, 326, 490])
        out = self.conv_block32_64(out) #torch.Size([2, 64, 86, 110, 78])
        # out = self.pool1(out) #torch.Size([2, 64, 43, 55, 39])
        # out = self.Dropout(out) #depending on % drop
        # print("32->64:", out.shape)
        out = self.strConv1(out)
        # print("str conv1:", out.shape)
        out = self.conv_block64_128(out) #torch.Size([2, 128, 43, 55, 39])
        # print("64->128:", out.shape)
        # out = self.pool2(out)
        out = self.strConv2(out)
        # print("str conv2:", out.shape)
        # out = self.Dropout(out)
        out = self.conv_block128_256(out)
        # print("128->256:", out.shape)
        # out = self.pool3(out)
        out = self.strConv3(out)
        # print("str conv3:", out.shape)
        # out = self.Dropout(out)
        # print("shape before fc:",out.shape) #torch.Size([1, 256, 11, 14, 10]) batchsize = 1
        # out = self.conv_block256_512(out) #if you use 256 in fc1 dont use this line
        # print('before reshape in Network:',out.shape) #torch.Size([2, 512, 11, 14, 10])
        # out = out.reshape(out.shape[0], -1) #flatten everything except batchsize, first dim
        # out = self.conv1d(out)
        out = out.reshape(out.shape[0],1,-1) #th middle 1 is the input channel of conv1d
        # out = out.reshape(out.shape[0], -1)
        # print("after flat: ", out.shape)
        # out = self.conv1d(out)
        # print('Here')
        # print("conv1D: ", out.shape) #torch.Size([4, 2, 650752])
        out = out.reshape(out.shape[0], -1)
        # print("again", out.shape)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out

class ConvBlock1(nn.Module): #use only for RFNet3D
    def __init__(self,in_channel,out_channel, activation=F.relu): #activation = F.relu
        super(ConvBlock1,self).__init__() #UNetConvBlock inherits the functionality of nn.Module
        self.conv1 = nn.Conv3d(in_channel,out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel,out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channel)
        # self.activation = F.tanh(self.out) #does it require input?
        self.activation = activation

        # init.xavier_uniform_(self.conv1.weight, gain=5/3)
        # init.constant_(self.conv1.bias,0)
        # init.xavier_uniform_(self.conv2.weight,gain=5/3)
        # init.constant_(self.conv2.bias,0)

    def forward(self, input):
        # output = torch.tanh(self.bn1(self.conv1(input)))
        output = self.activation(self.bn1(self.conv1(input)))
        # output =torch.tanh(self.bn2(self.conv2(output)))
        output = self.activation(self.bn2(self.conv2(output)))
        return output

class RFNet3D(nn.Module): #Network without any maxpool layer or residual connections...
    def __init__(self, in_channel = 1, n_classes=512, prob=0.2):
        super(RFNet3D,self).__init__()
        self.Dropout = nn.Dropout3d(p=prob)
        self.conv_block1_32 = ConvBlock1(in_channel, 32,1)
        self.conv_block32_64 = ConvBlock1(32,64,2)
        self.conv_block64_128 = ConvBlock1(64,128, 2)
        self.conv_block128_256 = ConvBlock1(128,256, 2)

        self.fc1 = nn.Linear(256*11*14*10, n_classes) #512 todo: get the hidden layer size
        self.fc2 = nn.Linear(n_classes,n_classes) #todo: tanh(), syntax gonna work?

    def forward(self, x): #torch.Size([2, 1, 86, 110, 78])
        out = self.conv_block1_32(x) #torch.Size([2, 32, 86, 110, 78])
        out = self.conv_block32_64(out) #torch.Size([2, 64, 86, 110, 78])
        #out = self.Dropout(out) #depending on % drop
        # print('1 drop')
        # print(type(out))
        out = self.conv_block64_128(out) #torch.Size([2, 128, 43, 55, 39])
        # print('post:',out.shape)
        #out = self.Dropout(out)
        out = self.conv_block128_256(out)
        #out = self.Dropout(out)
        # print('before reshape in Nerwork:',out.shape) #torch.Size([2, 512, 11, 14, 10])
        out = out.reshape(out.shape[0], -1) #flatten everything except batchsize, first dim
        out = self.fc1(out)
        out = self.fc2(out)

        return out
class Resnet_mob(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet_mob, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18_conv = nn.Sequential(*list(resnet18.children())[:-2])
        self.fc1 = nn.Linear(in_features= 512*11*16, out_features=2)
    def forward(self,x):
        x=self.resnet18_conv(x)
        x=x.view(x.shape[0], -1)
        x=self.fc1(x)
        # out = torch.sigmoid(x)
        return x

# if __name__ != "__main__":
#     input = torch.rand([2,1,86,110,78]).float().to(device) #if you run with gpu
#     # input = torch.rand([2, 1, 86, 110, 78])
#
# if __name__ != "__main__":
#     pool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2,padding=1,dilation=1)
#     # output = pool1(input)
#     conv1 = nn.Conv3d(32,64,kernel_size=3,stride=2,padding=1,dilation=2)
#
# if __name__ != "__main__":
#     model = ResNet3D().to(device) #cpu
#     # model = ResNet3D()
#     model.apply(weights_init)
#     output = model(input) #torch.Size([2, 32, 86, 110, 78]) -> torch.Size([2, 64, 43, 55, 39])
#     print(input.shape) #torch.Size([2, 32, 86, 110, 78])
#     print(output.shape) #torch.Size([2, 32, 43, 55, 39])
#     # print(model)

# if __name__ == "__main__":
#     torch.manual_seed(23)
#     model1 = ResNet3D()
#     model1.apply(weights_init)
#     torch.manual_seed(23)
#     model2 = ResNet3D()
#     model2.apply(weights_init)
#     # print(model1.conv_block128_256.conv2.weight == model2.conv_block128_256.conv2.weight)
#     print(model1.conv1d.conv1.bias == model2.conv1d.conv1.bias)
#     # print(model1.conv_block128_256.conv2.bias)
if __name__=='__main__':
    model = Resnet_mob(pretrained=True)
    print(model)
    # inp = torch.rand([1,3,224,224])
    # out=model(inp)
    # print(out.shape)