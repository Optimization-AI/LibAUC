import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

    
class FFNN(torch.nn.Module):
    r"""
    The basic 3-layer-MLP in multiple instance learning experiments utilized from [1]_.
    
    Args:
        input_dim (int, required): input data dimension.
        hidden_sizes (list[int], required): number of nurons for the hidden layers.
        num_class (int, required): number of class for model prediction, default: 1.
        last_activation (str, optional): the activation function for the output layer.
        
    Example:
        >>> model = FFNN_stoc_att(num_classes=1, dims=DIMS)

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://arxiv.org/abs/2305.08040
    """
    def __init__(self, input_dim=29, hidden_sizes=(16,), last_activation=None, num_classes=1):
        super(FFNN, self).__init__()
        self.inputs = torch.nn.Linear(input_dim, hidden_sizes[0])
        self.last_activation = last_activation
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.classifer = torch.nn.Linear(hidden_sizes[-1], num_classes)
    def forward(self, x):
        x = torch.tanh(self.inputs(x))
        x = self.layers(x)
        if self.last_activation is None:
            return self.classifer(x)
        elif self.last_activation == 'sigmoid':
            return torch.sigmoid(self.classifer(x))


class FFNN_stoc_att(nn.Module):
    r"""
    The basic 3-layer-MLP with an extra attention module that generates importance weights for combining the instance-level hidden features for each bag under multiple instance learning setting [1]_.

    Args:
        input_dim (int, required): input data dimension.
        hidden_sizes (list[int], required): number of nurons for the hidden layers.
        num_class (int, required): number of class for model prediction, default: 1.

    Example:
        >>> model = FFNN_stoc_att(num_classes=1, dims=DIMS)

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
           https://arxiv.org/abs/2305.08040
    """
    def __init__(self, input_dim=29, hidden_sizes=(16,), num_classes=1):
        super(FFNN_stoc_att, self).__init__()
        self.inputs = torch.nn.Linear(input_dim, hidden_sizes[0])
        layers = []
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.classifer = torch.nn.Linear(hidden_sizes[-1], num_classes)
        self.attention = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[-1], 1)
        )

        self.apply(_weights_init)
        print('model initialized')


    def forward(self, x):
        x = torch.tanh(self.inputs(x))
        x = self.layers(x)
        weights = self.attention(x)
        weights = torch.exp(weights)
        out = self.classifer(x)
        return out, weights



def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight) 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


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

    

class ResNet_stoc_att(nn.Module):
    r"""
    The ResNet [2,3,4]_ with an extra attention module that generates importance weights for combining the instance-level hidden features for each bag under multiple instance learning setting [1]_.

    Args:
        block (torch.nn.module, required): block module for ResNet.
        num_blocks (list[int], required): number of nurons for the hidden layer. 
        inchannels (int, required): the number of channels for the input image, default: 3.
        num_classes (int, required): the model prediction class number, default: 1.

    Example:
        >>> model = ResNet_stoc_att(block=BasicBlock, num_blocks=[3,3,3], inchannels=3, num_classes=1)

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023. https://arxiv.org/abs/2305.08040
           
        .. [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun "Deep Residual Learning for Image Recognition." arXiv:1512.03385
           
        .. [3] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
           
        .. [4] https://github.com/akamaster/pytorch_resnet_cifar10/tree/master
    """
    def __init__(self, block, num_blocks, inchannels=3, num_classes=1):
        super(ResNet_stoc_att, self).__init__()
        self.in_planes = 16
        self.inchannels = inchannels

        self.conv1 = nn.Conv2d(inchannels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.attention = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.bnlast = nn.BatchNorm1d(1)

    def init_weights(self):
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1,self.inchannels,x.shape[2],x.shape[3])
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, (out.size()[2],out.size()[3]))
        out = out.view(out.size()[0], -1)
        weights = self.attention(out)
        weights = torch.exp(weights)
        out = self.linear(out)
        return out, weights



def ResNet20_stoc_att(activations='relu', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return ResNet_stoc_att(BasicBlock, [3, 3, 3], **kwargs)


