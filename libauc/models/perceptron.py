import torch 
from torch import nn
import torch.nn.functional as F



class MLP(torch.nn.Module):
    r"""
        An implementation of Multilayer Perceptron (MLP).
    """
    def __init__(self, input_dim=29, hidden_sizes=(16,), activation='relu', num_classes=1):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        if sum(self.hidden_sizes) > 0: # multi-layer model
            self.inputs = torch.nn.Linear(input_dim, hidden_sizes[0]) 
            layers = []
            for i in range(len(hidden_sizes)-1):
                layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])) 
                if activation=='relu':
                  layers.append(nn.ReLU())
                elif activation=='elu':
                  layers.append(nn.ELU())
                else:
                  pass 
            self.layers = nn.Sequential(*layers)
            classifier_input_dim = hidden_sizes[-1]
        else: # linear model 
            classifier_input_dim = input_dim
        self.classifer = torch.nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x):
        """forward pass"""
        if sum(self.hidden_sizes) > 0:
            x = self.inputs(x)
            x = self.layers(x)
        return self.classifer(x) 
  
    
    
