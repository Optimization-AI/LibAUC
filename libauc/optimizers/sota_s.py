import math
import torch

class SOTAs(torch.optim.Optimizer):
    r"""
        Stochastic Optimization for Two-way pAUC Soft-version (SOTAs) is used for optimizing :obj:`~libauc.losses.tpAUC_KL_Loss`. The key update steps are summarized as follows:

            1. Initialize :math:`\mathbf u_0= \mathbf 0, v_0= \mathbf 0, \mathbf m_0= \mathbf 0, \mathbf w`
            2. For :math:`t=1, \ldots, T`:
            3. :math:`\hspace{5mm}` Sample two mini-batches :math:`\mathcal B_+\subset\mathcal S_+` and :math:`\mathcal B_-\subset\mathcal S_-`.
            4. :math:`\hspace{5mm}` For each :math:`\mathbf x_i\in\mathcal B_{+}`, update :math:`u^i_{t} =(1-\beta_0)u^i_{t-1} + \beta_0 \frac{1}{|B_-|}  \sum_{\mathbf x_j\in \mathcal B_-}L(\mathbf w_t; \mathbf x_i, \mathbf x_j)`
            5. :math:`\hspace{5mm}` Update :math:`v_{t} = (1-\beta_1)v_{t-1} + \beta_1\frac{1}{|\mathcal B_{+}|}\sum_{\mathbf x_i\in \mathcal B_{+}} f_2(u^i_{t-1})`
            6. :math:`\hspace{5mm}` Compute :math:`p_{ij} = (u^i_{t-1})^{\lambda/\lambda' - 1}\exp (L(\mathbf w_t, \mathbf x_i, \mathbf x_j)/\lambda)/v_{t}`
            7. :math:`\hspace{5mm}` Compute a gradient estimator:
            
                .. math::
                    \nabla_t=\frac{1}{|\mathcal B_{+}}\frac{1}{|\mathcal B_-|}\sum_{\mathbf x_i\in\mathcal B_{+}}   \sum_{\mathbf x_j\in \mathcal B_-}p_{ij}\nabla L(\mathbf w_t; \mathcal x_i, \mathcal x_j)

            8. :math:`\hspace{5mm}` Compute :math:`\mathbf m_{t}=(1-\beta_2)\mathbf m_{t-1} + \beta_2 \nabla_t`
            9. :math:`\hspace{5mm}` Update :math:`\mathbf w_{t+1}   =\mathbf w_t - \eta_1  \mathbf m_t` (or Adam style)
            
            
        For more details, please refer to the paper `When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee <https://proceedings.mlr.press/v162/zhu22g.html>`__.

        Args:
            params (iterable): iterable of parameters to optimize
            lr (float, optional): learning rate (default: ``0.1``)
            mode (str, optional): optimization mode, 'sgd' or 'adam' (default: ``'sgd'``)
            weight_decay (float, optional): weight decay (L2 penalty) (default: ``1e-5``)
            epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``0.0``)
            momentum (float, optional): momentum factor for 'sgd' mode (default: ``0.9``)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square for 'adam' mode. (default: ``(0.9, 0.999)``)
            eps (float, optional): term added to the denominator to improve
                numerical stability for 'adam' mode (default: ``1e-8``)
            amsgrad (bool, optional): whether to use the AMSGrad variant of 'adam' mode
                from the paper `On the Convergence of Adam and Beyond` (default: ``False``)
            verbose (bool, optional): whether to print optimization progress (default: ``True``)
            device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``None``)

        Example:
            >>> optimizer = libauc.optimizers.SOTAs(model.parameters(), loss_fn=loss_fn, lr=0.1, momentum=0.9)
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
                 mode='adam',
                 lr=1e-3, 
                 clip_value=1.0,
                 weight_decay=0, 
                 epoch_decay=0,
                 betas=(0.9, 0.999), 
                 eps=1e-8,  
                 amsgrad=False, 
                 momentum=0.9, 
                 nesterov=False, 
                 dampening=0,                 
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
            
        self.params = list(params) # support optimizing partial parameters of models
        self.lr = lr

        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None
        self.T = 0                # for epoch_decay
        self.steps = 0            # total optimization steps
        self.verbose = verbose    # print updates for lr/regularizer

        self.epoch_decay = epoch_decay
        self.mode = mode.lower()
        assert self.mode in ['adam', 'sgd'], "Keyword is not found in [`adam`, `sgd`]!"
        
        defaults = dict(lr=lr, betas=betas, eps=eps, momentum=momentum, nesterov=nesterov, dampening=dampening, 
                        epoch_decay=epoch_decay, weight_decay=weight_decay, amsgrad=amsgrad,
                        clip_value=clip_value, model_ref=self.model_ref, model_acc=self.model_acc)
        super(SOTAs, self).__init__(self.params, defaults)

        
    def __setstate__(self, state):
      r"""
      # Set default options for sgd mode and adam mode
      """
      super(SOTAs, self).__setstate__(state)
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
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
              
            epoch_decay = group['epoch_decay']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            clip_value = group['clip_value']
            weight_decay = group['weight_decay']
            
            if self.mode == 'adam':
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
                  #if group['weight_decay'] != 0:
                  #    grad = grad.add(p, alpha=group['weight_decay'])
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
                       
            elif self.mode == 'sgd':
              for i, p in enumerate(group['params']):
                  if p.grad is None:
                      continue
                  #d_p = p.grad
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
                  
        self.steps += 1  
        self.T += 1   
        return loss
    
    def update_lr(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
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