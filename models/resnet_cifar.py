# This implementation is adapted from https://github.com/akamaster/pytorch_resnet_cifar10/tree/master.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
         #init.kaiming_normal_(m.weight)
         init.xavier_normal_(m.weight) 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
from torch.nn import Parameter
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = activation_func(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, last_activation='sigmoid', pretrained=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        self.last_activation = last_activation
        if self.last_activation is not None:
            self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        else:
            out = self.sigmoid(out)
        return out


def resnet20(pretrained=False, activations='relu', last_activation=None, **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    # print (activation_func)
    return ResNet(BasicBlock, [3, 3, 3], last_activation=last_activation, **kwargs)


def resnet32(pretrained=False, activations='relu', last_activation=None, **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    # print (activation_func)
    return ResNet(BasicBlock, [5, 5, 5], last_activation=last_activation, **kwargs)


def resnet44(pretrained=False, activations='relu', last_activation=None, **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    # print (activation_func)    
    return ResNet(BasicBlock, [7, 7, 7], last_activation=last_activation, **kwargs)


def resnet56(pretrained=False, activations='relu',  last_activation=None, **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    # print (activation_func)
    return ResNet(BasicBlock, [9, 9, 9], last_activation=last_activation, **kwargs)


def resnet110(pretrained=False, activations='relu', last_activation=None, **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    # print (activation_func)
    return ResNet(BasicBlock, [18, 18, 18], last_activation=last_activation, **kwargs)


def resnet1202(pretrained=False, activations='relu', last_activation=None, **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    # print (activation_func)
    return ResNet(BasicBlock, [200, 200, 200], last_activation=last_activation, **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

    
# alias 
ResNet20 = resnet20
ResNet32 = resnet32 
ResNet44 = resnet44
ResNet56 = resnet56
ResNet110 = resnet110 
ResNet1202 = resnet1202

if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
