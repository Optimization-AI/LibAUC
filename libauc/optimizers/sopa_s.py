import torch
import math

class SOPAs(torch.optim.Optimizer):
    r"""
        Stochastic Optimization for One-way pAUC (SOPAs) is used for optimizing :obj:`~libauc.losses.pAUC_DRO_Loss`. The key update steps are summarized as follows:

            1. Initialize :math:`\mathbf u^0=0, \mathbf w_0`
            2. For :math:`t=1, \ldots, T`:
            3. :math:`\hspace{5mm}` For each :math:`\mathbf{x}_i\in \mathbf{B}_{+}`, update :math:`u^{t}_i =(1-\gamma)u^{t-1}_{i} + \gamma \frac{1}{|\mathbf{B}_-|}  \sum_{\mathbf{x}_j\in \mathbf{B}_-}\exp\left(\frac{L(\mathbf{w}_t; \mathbf{x}_i, \mathbf{x}_j)}{\lambda}\right)`
            4. :math:`\hspace{5mm}` Let :math:`p_{ij} = \exp (L(\mathbf{w}_t; \mathbf{x}_i, \mathbf{x}_j)/\lambda)/u^{t}_{i}`, then compute a gradient estimator:
            
                :math:`\nabla_t=\frac{1}{|\mathbf{B}_{+}|}\frac{1}{|\mathbf{B}_-|}\sum_{\mathbf{x}_i\in\mathbf{B}_{+}}   \sum_{\mathbf{x}_j\in \mathbf{B}_-}p_{ij}\nabla L(\mathbf{w}_t; \mathbf{x}_i, \mathbf{x}_j)`
         
            5. :math:`\hspace{5mm}` Update :math:`\mathbf{v}_{t}=\beta\mathbf{v}_{t-1} + (1-\beta) \nabla_t`
            6. :math:`\hspace{5mm}` Update :math:`\mathbf{w}_{t+1}=\mathbf{w}_t - \eta  \mathbf{v}_t` (or Adam-style)
            
            
        For more details, please refer to `When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee <https://proceedings.mlr.press/v162/zhu22g.html>`__.

        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate (default: ``1e-3``)
            mode (str, optional): optimization mode, 'sgd' or 'adam' (default: ``'adam'``)
            clip_value (float, optional): gradient clipping value (default: ``2.0``)
            weight_decay (float, optional): weight decay (L2 penalty) (default: ``1e-5``)
            epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``0.0``)
            momentum (float, optional): momentum factor (default: ``0.9``)
            dampening (float, optional): dampening for momentum (default: ``0.1``)
            nesterov (bool, optional): enables Nesterov momentum (default: ``False``)
            betas (Tuple[float, float], optional): coefficients used for computing
                    running averages of gradient and its square (default: ``(0.9, 0.999)``)
            eps (float, optional): term added to the denominator to improve
                    numerical stability (default: ``1e-8``)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                    algorithm from the paper `On the Convergence of Adam and Beyond`_
                    (default: ``False``)
            verbose (bool, optional): whether to print optimization progress (default: ``True``)
            device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``None``)

        Example:
            >>> optimizer = libauc.optimizers.SOPAs(model.parameters(), lr=1e-3, mode='adam')
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()

        Reference:
            .. [1] Zhu, Dixian and Li, Gang and Wang, Bokun and Wu, Xiaodong and Yang, Tianbao.
               "When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee."
               In International Conference on Machine Learning, pp. 27548-27573. PMLR, 2022.
               https://proceedings.mlr.press/v162/zhu22g.html
    """
    
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 mode = 'adam',
                 clip_value=2.0,
                 weight_decay=1e-5, 
                 epoch_decay=0,
                 momentum=0.9, 
                 nesterov=False, 
                 dampening=0.1, 
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
           
        self.params = list(params) # support optimizing partial parameters of models
        self.lr = lr
        self.mode = mode.lower()
        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None
        self.T = 0                # for epoch_decay
        self.steps = 0            # total optimization steps
        self.verbose = verbose    # print updates for lr/regularizer
        self.steps = 0
        self.epoch_decay = epoch_decay

        assert self.mode in ['adam', 'sgd'], "Keyword is not found in [`adam`, `sgd`]!"

        defaults = dict(lr=lr, weight_decay=weight_decay, epoch_decay=epoch_decay,
                        momentum=momentum, dampening=dampening, nesterov=nesterov,
                        betas=betas, eps=eps, amsgrad=amsgrad, 
                        clip_value=clip_value, model_ref=self.model_ref, model_acc=self.model_acc)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super(SOPAs, self).__init__(self.params, defaults)
        

    def __setstate__(self, state):
      r"""
      # Set default options for sgd mode and adam mode
      """
      super(SOPAs, self).__setstate__(state)
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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
             
            clip_value = group['clip_value']
            epoch_decay = group['epoch_decay']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
             
            if self.mode == 'sgd':
               for i, p in enumerate(group['params']):
                  if p.grad is None:
                      print(p.shape)
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
                  p.add_(d_p, alpha=-group['lr'])
                  if epoch_decay > 0:
                     model_acc[i].data = model_acc[i].data + p.data

            elif self.mode == 'adam':
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

                    step_size = group['lr'] / bias_correction1

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if epoch_decay > 0:
                       model_acc[i].data = model_acc[i].data + p.data
                    
        self.steps += 1
        self.T += 1  
        return loss

    def update_lr(self, decay_factor=None):
        if decay_factor != None:
           self.param_groups[0]['lr'] =  self.param_groups[0]['lr']/decay_factor
           print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))

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