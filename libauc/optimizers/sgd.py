import torch
from torch.optim.optimizer import Optimizer, required 

class SGD(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum). This code is adapted from `PyTorch codebase <https://github.com/pytorch/pytorch/blob/v1.2.0/torch/optim/sgd.py>`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize 
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: ``0``)
        weight_decay (float, optional): weight decay (L2 penalty) (default: ``0``)
        epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``0.0``)
        dampening (float, optional): dampening for momentum (default: ``0.0``)
        nesterov (bool, optional): enables Nesterov momentum (default: ``False)``
        device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``None``).

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, 
                 params, 
                 lr=required, 
                 momentum=0, 
                 dampening=0,
                 clip_value=1.0, 
                 epoch_decay=0,
                 weight_decay=0, 
                 nesterov=False,
                 verbose=True,            
                 device=None,
                 **kwargs):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:  
            self.device = device
            
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
           
        self.params = list(params)
        self.lr = lr

        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None

        self.T = 0                # for epoch_decay
        self.steps = 0            # total optim steps
        self.verbose = verbose    # print updates for lr/regularizer
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, epoch_decay=epoch_decay, nesterov=nesterov,
                        clip_value=clip_value, model_ref=self.model_ref, model_acc=self.model_acc)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(self.params, defaults)
        
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    def __init_model_ref__(self, params):
         model_ref = []
         if not isinstance(params, list):
            params = list(params)
         for var in params: 
            if var is not None:
               model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
         return model_ref
     
    def __init_model_acc__(self, params):
        model_acc = []
        if not isinstance(params, list):
           params = list(params)
        for var in params: 
            if var is not None:
               model_acc.append(torch.zeros(var.shape, dtype=torch.float32,  device=self.device, requires_grad=False).to(self.device)) 
        return model_acc
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            epoch_decay = group['epoch_decay']
            clip_value = group['clip_value']
            weight_decay = group['weight_decay']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
        
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if epoch_decay > 0:
                    d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                else:
                    d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + weight_decay*p.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-self.lr)
                if epoch_decay > 0:
                   model_acc[i].data = model_acc[i].data + p.data
                
        self.T += 1  
        self.steps += 1
        return loss
    
    def update_lr(self, decay_factor=None):
        r"""Updates learning rate given a decay factor."""
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f !'%(self.param_groups[0]['lr']))
    
    def update_regularizer(self, decay_factor=None):
        r"""Updates learning rate given a decay factor and resets epoch-decay regularizer."""
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if self.verbose:    
           print ('Updating regularizer @ T=%s!'%(self.steps))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0
