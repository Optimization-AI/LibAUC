import torch 
import math 

class PESG(torch.optim.Optimizer):
    r"""
        Proximal Epoch Stochastic Gradient Method (PESG) is used for optimizing the :obj:`~libauc.losses.AUCMLoss`. The key update steps are summarized as follows:

            1. Initialize :math:`\mathbf v_0= \mathbf v_{ref}=\{\mathbf{w_0}, a_0, b_0\}, \alpha_0\geq 0`
            2. For :math:`t=1, \ldots, T`:
            3. :math:`\hspace{5mm}` Compute :math:`\nabla_{\mathbf v} F(\mathbf v_t, \alpha_t; z_t)` and :math:`\nabla_\alpha F(\mathbf v_t, \alpha_t; z_t)`.
            4. :math:`\hspace{5mm}` Update primal variables
            
                .. math::
                    \mathbf v_{t+1} = \mathbf v_{t} - \eta (\nabla_{\mathbf v} F(\mathbf v_t, \alpha_t; z_t)+ \lambda_0 (\mathbf v_t-\mathbf v_{\text{ref}})) - \lambda \eta\mathbf v_t 

            5. :math:`\hspace{5mm}` Update dual variable

                .. math::
                    \alpha_{t+1}=  [\alpha_{t} + \eta \nabla_\alpha F(\mathbf v_t, \alpha_t; z_t)]_+

            6. :math:`\hspace{5mm}` Decrease :math:`\eta` by a decay factor and update :math:`\mathbf v_{\text{ref}}` periodically

            where :math:`z_t` is the data pair :math:`(x_t, y_t)`, :math:`\lambda_0` is the epoch-level l2 penalty (i.e., `epoch_decay`), :math:`\lambda` is the l2 penalty (i.e., `weight_decay`), 
            and :math:`\eta` is the learning rate. 

            For more details, please refer to the paper `Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification <https://arxiv.org/abs/2012.03173>`__.

        Args:
            params (iterable): iterable of parameters to optimize
            loss_fn (callable): loss function used for optimization (default: ``None``)
            lr (float): learning rate (default: ``0.1``)
            mode (str): optimization mode, 'sgd' or 'adam' (default: ``'sgd'``)
            clip_value (float, optional): gradient clipping value (default: ``1.0``)
            weight_decay (float, optional): weight decay (L2 penalty) (default: ``1e-5``)
            epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``2e-3``)
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
            >>> optimizer = libauc.optimizers.PESG(model.parameters(), lr=0.1, momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()


        Reference:
            .. [1] Yuan, Zhuoning, Yan, Yan, Sonka, Milan, and Yang, Tianbao.
               "Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification."
               Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
               https://arxiv.org/abs/2012.03173
    """
    def __init__(self, 
                 params, 
                 loss_fn,
                 lr=0.1, 
                 mode='sgd',
                 clip_value=1.0, 
                 weight_decay=1e-5, 
                 epoch_decay=2e-3, 
                 momentum=0.9,
                 betas=(0.9, 0.999), 
                 eps=1e-8, 
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
        if not 0.0 <= epoch_decay:
            raise ValueError("Invalid epoch_decay value: {}".format(epoch_decay))  
        if not isinstance(mode, str):
           raise ValueError("Invalid mode type: {}".format(mode))

        self.params = list(params)
        self.lr = lr
        self.mode = mode.lower()
        self.clip_value = clip_value
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epoch_decay = epoch_decay
        self.loss_fn = loss_fn
        self.a = None
        self.b = None
        self.alpha = None
        self.margin = None

        try:
            self.a = loss_fn.a 
            self.b = loss_fn.b 
            self.alpha = loss_fn.alpha 
            self.margin = loss_fn.margin
        except:
            print('AUCMLoss is not found!')

        # init 
        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None
        self.T = 0                # for epoch_decay
        self.steps = 0            # total optimization steps
        self.verbose = verbose    # print updates for lr/regularizer
    
        assert self.mode in ['adam', 'sgd'], "Keyword is not found in [`adam`, `sgd`]!"
       
        if self.a is not None and self.b is not None:
           self.params = self.params + [self.a, self.b]

        self.defaults = dict(lr=self.lr, 
                             margin=self.margin, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=clip_value,
                             momentum=momentum,
                             betas=betas, 
                             eps=eps, 
                             amsgrad=amsgrad,
                             weight_decay=weight_decay,
                             epoch_decay=epoch_decay,
                             model_ref=self.model_ref,
                             model_acc=self.model_acc)
        
        super(PESG, self).__init__(self.params, self.defaults)
         
    def __setstate__(self, state):
        super(PESG, self).__setstate__(state)
        for group in self.param_groups:
          if self.mode == 'sgd':
             group.setdefault('nesterov', False)
          elif self.mode == 'adam':
             group.setdefault('amsgrad', False)
          else:
             NotImplementedError    

    def __init_model_ref__(self, params):
         model_ref = []
         if not isinstance(params, list):
            params = list(params)
         for var in params + [self.a, self.b]: 
            if var is not None:
               model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
         return model_ref
     
    def __init_model_acc__(self, params):
        model_acc = []
        if not isinstance(params, list):
           params = list(params)
        for var in params + [self.a, self.b]: 
            if var is not None:
               model_acc.append(torch.zeros(var.shape, dtype=torch.float32,  device=self.device, requires_grad=False).to(self.device)) 
        return model_acc
    
    @property    
    def optim_step(self):
        r"""Return the number of optimization steps."""
        return self.steps
    
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
            weight_decay = group['weight_decay']
            clip_value = group['clip_value']
            momentum = group['momentum']
            self.lr =  group['lr']
            
            epoch_decay = group['epoch_decay']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            
            m = group['margin'] 
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            if self.mode == 'sgd':
                # updates
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
                            buf.mul_(1-momentum).add_(d_p, alpha=momentum)
                        d_p =  buf
                    p.data = p.data - group['lr']*d_p
                    if epoch_decay > 0:
                       model_acc[i].data = model_acc[i].data + p.data
            elif self.mode == 'adam':
                # updates
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    if epoch_decay > 0:
                        grad = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                    else:
                        grad = torch.clamp(p.grad.data , -clip_value, clip_value) + weight_decay*p.data
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

                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if epoch_decay > 0:
                       model_acc[i].data = model_acc[i].data + p.data

            if alpha is not None: 
               if alpha.grad is not None: 
                  alpha.data = alpha.data + group['lr']*(2*(m + b.data - a.data)-2*alpha.data)
                  #alpha.data = alpha.data + group['lr']*alpha.grad 
                  alpha.data  = torch.clamp(alpha.data,  0, 999)

        self.T += 1  
        self.steps += 1
        return loss

        
    def update_lr(self, decay_factor=None):
        r"""Updates learning rate given a decay factor."""
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
            
    def update_regularizer(self, decay_factor=None):
        r"""Updates learning rate given a decay factor and resets epoch-level regularizer."""
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
        
  
