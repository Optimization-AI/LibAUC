import torch
import math
from torch.optim.optimizer import required 

class iSogCLR(torch.optim.Optimizer):
    r"""
    Stochastic Optimization for sovling :obj:`~libauc.losses.GCLoss`. For each iteration **t**, the key updates for **iSogCLR** are sumarized as follow:
        
        1. Initialize :math:`\mathbf w_1, \mathbf{\tau}=\tau_{\text{init}}, \mathbf s_1 = \mathbf v_1 = \mathbf u_1= \mathbf 0`
        2. For :math:`t=1, \ldots, T`:
        3. :math:`\hspace{5mm}` Draw a batch of :math:`B` samples
        4. :math:`\hspace{5mm}` For :math:`\mathbf{x}_i \in \mathbf{B}`:
        5. :math:`\hspace{10mm}` Compute :math:`g_i (\mathbf{w_t}, \mathbf{\tau}_i^t; \mathbf{B}_i) = \frac{1}{B} \sum_{z\in\mathbf{B}_i)} \exp \left(\frac{h_i(z)}{\mathbf{\tau}_i^t}  \right)`
        6. :math:`\hspace{10mm}` Update :math:`\mathbf{s}_i^{t+1} = (1-\beta_0) \mathbf{s}_i^{t} + \beta_0 g_i (\mathbf{w_t}, \mathbf{\tau}_i^t; \mathbf{B}_i)`
        7. :math:`\hspace{10mm}` Compute :math:`G(\mathbf{\tau}_i^t) = \frac{1}{n} \left[\frac{\mathbf{\tau}_i^t}{\mathbf{s}_i^t} \nabla_{\mathbf{\tau}_i} g_i (\mathbf{w_t}, \mathbf{\tau}_i^t; \mathbf{B}_i) + \log(\mathbf{s}_i^t) + \rho  \right]`
        8. :math:`\hspace{10mm}` Update :math:`\mathbf{u}_i^{t+1} = (1-\beta_1) \mathbf{u}_i^{t} + \beta_1 G(\mathbf{\tau}_i^t)`
        9. :math:`\hspace{10mm}` Update :math:`\mathbf{\tau}_i^{t+1} = \Pi_{\Omega}[\mathbf{\tau}_i^{t} - \eta \mathbf{u}_i^{t+1}]`
        10. :math:`\hspace{5mm}` Compute stochastic gradient estimator :math:`G(\mathbf{w}_t) = \frac{1}{B} \sum_{\mathbf{x}_i \in \mathbf{B}} \frac{\mathbf{\tau}_i^t}{\mathbf{s}_i^t} \nabla_{\mathbf{w}} g_i (\mathbf{w_t}, \mathbf{\tau}_i^t; \mathbf{B}_i)`
        11. :math:`\hspace{5mm}` Update model :math:`\mathbf{w_t}` by *Momemtum* or *Adam optimzier*

    where :math:`h_i(z)=E(\mathcal{A}(\mathbf{x}_i))^{\top} E(z) - E(\mathcal{A}(\mathbf{x}_i))^{\top} E(\mathcal{A}^{\prime}(\mathbf{x}_i))`, :math:`\mathbf{B}_i = \{\mathcal{A}(\mathbf{x}), \mathcal{A}^{\prime}(\mathbf{x}): \mathcal{A},\mathcal{A}^{\prime}\in\mathcal{P},\mathbf{x}\in \mathbf{B} \backslash \mathbf{x}_i \}`,
    :math:`\Omega=\{\tau_0 \leq \tau \}` is the constraint set for each learnable :math:`\mathbf{\tau}_i`, :math:`\Pi` is the projection operator.  

    For more details, please refer to `Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization <https://arxiv.org/abs/2305.11965>`__.
    
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: ``0.1``)
        mode (str): optimization mode, 'sgd' or 'adam' (default: ``'sgd'``)
        weight_decay (float, optional): weight decay (L2 penalty) (default: ``1e-5``)
        epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``0.0``)
        momentum (float, optional): momentum factor for 'sgd' mode (default: ``0.9``)
        betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square for 'adam' mode (default: ``(0.9, 0.999)``)
        eps (float, optional): term added to the denominator to improve
                numerical stability for 'adam' mode (default: ``1e-8``)
        amsgrad (bool, optional): whether to use the AMSGrad variant of 'adam' mode
                from the paper `On the Convergence of Adam and Beyond` (default: ``False``)
        verbose (bool, optional): whether to print optimization progress (default: ``True``)
        device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``None``)

    Example:
            >>> optimizer = libauc.optimizers.iSogCLR(model.parameters(),lr=0.1, mode='lars', momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target, index).backward()
            >>> optimizer.step()

    """
    def __init__(self, 
                 params, 
                 lr=required, 
                 clip_value=10.0, 
                 weight_decay=1e-6, 
                 epoch_decay=0, 
                 mode='lars', 
                 momentum=0, 
                 trust_coefficient=0.001,  
                 betas=(0.9, 0.999), 
                 eps=1e-8, 
                 amsgrad=False,
                 verbose=False,
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
        if not isinstance(mode, str):
           raise ValueError("Invalid mode type: {}".format(mode))
                
        self.params = list(params)
        self.lr = lr
        self.mode = mode.lower()
        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None
        self.T = 0                # for epoch_decay
        self.steps = 0            # total optimization steps
        self.verbose = verbose    # print updates for lr/regularizer
        assert self.mode in ['adamw', 'lars'], "Keyword is not found in [`adamw`, `lars`]!"
        
        defaults = dict(lr=lr, clip_value=clip_value, weight_decay=weight_decay, epoch_decay=epoch_decay, 
                        momentum=momentum, trust_coefficient=trust_coefficient,
                        betas=betas, eps=eps, amsgrad=amsgrad, model_ref=self.model_ref, model_acc=self.model_acc)
        
        super(iSogCLR, self).__init__(self.params, defaults)
            
    def __setstate__(self, state):
        super(iSogCLR, self).__setstate__(state)
        for group in self.param_groups:
            if self.mode == 'adamw':
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

            if self.mode == 'lars':
                for i, p in enumerate(group['params']):
                    dp = p.grad

                    if dp is None:
                        continue

                    if p.ndim > 1: # if not normalization gamma/beta or bias
                        dp = dp.add(p, alpha=group['weight_decay'])
                        # add epoch decay
                        if epoch_decay > 0:
                            dp = dp +  epoch_decay*(p.data - model_ref[i].data)
                        param_norm = torch.norm(p)
                        update_norm = torch.norm(dp)
                        one = torch.ones_like(param_norm)
                        q = torch.where(param_norm > 0.,
                                        torch.where(update_norm > 0,
                                        (group['trust_coefficient'] * param_norm / update_norm), one),
                                        one)
                        dp = dp.mul(q)

                    param_state = self.state[p]
                    if 'mu' not in param_state:
                        param_state['mu'] = torch.zeros_like(p)
                    mu = param_state['mu']
                    mu.mul_(group['momentum']).add_(dp)
                    p.add_(mu, alpha=-group['lr'])
                    if epoch_decay > 0:
                       model_acc[i].data = model_acc[i].data + p.data

            else:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    # Perform stepweight decay
                    p.mul_(1 - self.lr * group['weight_decay'])

                    # Perform optimization step
                    if epoch_decay > 0:
                        grad = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data)  
                    else:
                        grad = torch.clamp(p.grad.data , -clip_value, clip_value)

                    if grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
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
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = self.lr / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if epoch_decay > 0:
                       model_acc[i].data = model_acc[i].data + p.data 
        return loss


    def update_regularizer(self, decay_factor=None):
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
