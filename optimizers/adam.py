import math
import torch

class Adam(torch.optim.Optimizer):
    r"""Implements Adam algorithm. This code is adapted from `PyTorch codebase <https://github.com/pytorch/pytorch/blob/v1.2.0/torch/optim/adam.py>`__.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: ``1e-3``)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: ``(0.9, 0.999)``)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: ``1e-8``)
        weight_decay (float, optional): weight decay (L2 penalty) (default: ``0``)
        epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``0.0``)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: ``False``)
        device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``None``)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 clip_value=1.0,
                 epoch_decay=0,
                 weight_decay=0, 
                 amsgrad=False, 
                 verbose=True,
                 device=None,
                 **kwargs):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:  
            self.device = device
            
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            
        self.params = list(params)
        self.lr = lr

        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None

        self.T = 0                # for epoch_decay
        self.steps = 0            # total optim steps
        self.verbose = verbose    # print updates for lr/regularizer
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, epoch_decay=epoch_decay, amsgrad=amsgrad, 
                        clip_value=clip_value, model_ref=self.model_ref, model_acc=self.model_acc)
        super(Adam, self).__init__(self.params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            
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
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            epoch_decay = group['epoch_decay']
            clip_value = group['clip_value']
            weight_decay = group['weight_decay']
    
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if epoch_decay > 0:  
                    grad = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                else:
                    grad = torch.clamp(p.grad.data , -clip_value, clip_value) + weight_decay*p.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = self.lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
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
