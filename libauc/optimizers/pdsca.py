import torch 
import copy

class PDSCA(torch.optim.Optimizer):
    r"""
        Primal-Dual Stochastic Compositional Adaptive Algorithm (PDSCA) is used for optimizing :obj:`~libauc.losses.CompositionalAUCLoss`. For itearton :math:`t`, the key update steps are summarized as follows:
            
            1. Initialize :math:`\mathbf v_0= \mathbf v_{ref}=\mathbf u_0= \{\mathbf{w_0}, a_0, b_0\}, \alpha_0 \geq 0` 
            2. For :math:`t=1, \ldots, T`:
            3. :math:`\hspace{5mm} \mathbf{u}_{t+1}=(1-\beta_{0}) \mathbf{u}_{t}+\beta_{0}(\mathbf{w}_{\mathbf{t}}-\eta_0 \nabla L_{CE}(\mathbf{w}_{\mathbf{t}}) ; a ; b)`
            4. :math:`\hspace{5mm}` :math:`\mathbf{z}_{t+1}=(1-\beta_{1}) \mathbf{z}_{t}+\beta_{1} \nabla_{\mathbf{u}} L_{AUC}(\mathbf{u}_{t+1})`
            5. :math:`\hspace{5mm}` :math:`\mathbf{v}_{t+1}=\mathbf{v}_{t}-\eta_{1} (\mathbf{z}_{t+1} + λ_0(\mathbf{w}_t-\mathbf{v}_{ref})+ λ_1\mathbf{v}_t)`
            6. :math:`\hspace{5mm}` :math:`\theta_{t+1}=\theta_{t}+\eta_{1} \nabla_{\theta} L_{AUC}(\theta_{t})`
            7. :math:`\hspace{5mm}` Decrease :math:`\eta_0, \eta_1` by a decay factor and update :math:`\mathbf v_{\text{ref}}` periodically

        where :math:`\lambda_0,\lambda_1` refer to ``epoch_decay`` and ``weight_decay``, :math:`\eta_0, \eta_1` refer to learning rates for inner updates (:math:`L_{CE}`) and 
        outer updates (:math:`L_{AUC}`), and :math:`\mathbf v_t` refers to :math:`\{\mathbf w_t, a_t, b_t\}` and :math:`\theta` refers to dual variable in :obj:`CompositionalAUCLoss`. 
        For more details, please refer to `Compositional Training for End-to-End Deep AUC Maximization <https://openreview.net/pdf?id=gPvB4pdu_Z>`__.
    
        Args:
           params (iterable): iterable of parameters to optimize.
           loss_fn (callable): loss function used for optimization (default: ``None``)
           lr (float): learning rate (default: ``0.1``)
           lr0 (float, optional): learning rate for inner updates (default: ``None``)
           beta1 (float, optional): coefficient for updating the running average of gradient (default: ``0.99``)
           beta2 (float, optional): coefficient for updating the running average of gradient square (default: ``0.999``)
           clip_value (float, optional): gradient clipping value (default: ``1.0``)
           weight_decay (float, optional): weight decay (L2 penalty) (default: ``1e-5``).
           epoch_decay (float, optional): epoch decay (epoch-wise l2 penalty) (default: ``2e-3``)
           verbose (bool, optional): whether to print optimization progress (default: ``True``)
           device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``'cuda'``)

        Example:
            >>> optimizer = libauc.optimizers.PDSCA(model.parameters(), lr=0.1, momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()


        Reference:
            .. [1] Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang.
               "Compositional Training for End-to-End Deep AUC Maximization."
               International Conference on Learning Representations. 2022.
               https://openreview.net/forum?id=gPvB4pdu_Z
    """
    def __init__(self, 
                 params, 
                 loss_fn,
                 lr=0.1, 
                 lr0=None,
                 beta1=0.99,
                 beta2=0.999,
                 clip_value=1.0, 
                 weight_decay=1e-5, 
                 epoch_decay=2e-3, 
                 verbose=True,
                 device='cuda',
                 **kwargs):

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:  
            self.device = device      

        if lr0 is None:
           lr0 = lr
        self.lr = lr
        self.lr0 = lr0
        self.clip_value = clip_value
        self.weight_decay = weight_decay
        self.epoch_decay = epoch_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.loss_fn = loss_fn
        self.params = list(params)

        self.a = None
        self.b = None
        self.alpha = None
        self.margin = None

        if loss_fn != None:
            try:
                self.a = loss_fn.a 
                self.b = loss_fn.b 
                self.alpha = loss_fn.alpha 
                self.margin = loss_fn.margin
            except:
                print('CompositionalAUCLoss is not found!')
        else:
            raise ValueError('No loss_fn found!')

            
        self.model_ref = self.__init_model_ref__(self.params) if epoch_decay > 0 else None
        self.model_acc = self.__init_model_acc__(self.params) if epoch_decay > 0 else None
        self.T = 0                # for epoch_decay
        self.steps = 0            # total optim steps
        self.verbose = verbose    # print updates for lr/regularizer

        # TODO: if I can use model.parameters and remove model as input (to simplify)
        if self.a is not None and self.b is not None:
           self.params = self.params + [self.a, self.b]

        self.defaults = dict(lr=self.lr, 
                             lr0=self.lr0,
                             margin=self.margin, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=self.clip_value,
                             weight_decay=self.weight_decay,
                             epoch_decay=self.epoch_decay,
                             beta1=self.beta1,
                             beta2=self.beta2,
                             model_ref=self.model_ref,
                             model_acc=self.model_acc)
        
        super(PDSCA, self).__init__(self.params, self.defaults)

    def __setstate__(self, state):
        super(PDSCA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            self.lr =  group['lr']
            self.lr0 = group['lr0']
            
            epoch_decay = group['epoch_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            
            m = group['margin']
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            for i, p in enumerate(group['params']):
                if p.grad is None: 
                   continue
                if epoch_decay > 0:
                    d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                else:
                    d_p = torch.clamp(p.grad.data , -clip_value, clip_value)  + weight_decay*p.data
                if alpha.grad is None: # sgd + moving g
                    p.data = p.data - group['lr0']*d_p 
                    if beta1!= 0: 
                        param_state = self.state[p]
                        if 'weight_buffer' not in param_state:
                            buf = param_state['weight_buffer'] = torch.clone(p).detach()
                        else:
                            buf = param_state['weight_buffer']
                            buf.mul_(1-beta1).add_(p, alpha=beta1)
                        p.data =  buf.data # Note: use buf(s) to compute the gradients w.r.t AUC loss can lead to a slight worse performance 
                elif alpha.grad is not None: # auc + moving g
                   if beta2!= 0: 
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(1-beta2).add_(d_p, alpha=beta2)
                        d_p =  buf
                   p.data = p.data - group['lr']*d_p 
                else:
                    NotImplementedError
                if epoch_decay > 0: 
                   model_acc[i].data = model_acc[i].data + p.data
                
            if alpha is not None: 
               if alpha.grad is not None: 
                  alpha.data = alpha.data + group['lr']*(2*(m + b.data - a.data)-2*alpha.data)
                  alpha.data  = torch.clamp(alpha.data,  0, 999)
              
        self.T += 1        
        self.steps += 1
        return loss
        
    def update_lr(self, decay_factor=None, decay_factor0=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'],  self.steps))
        if decay_factor0 != None:
            self.param_groups[0]['lr0'] = self.param_groups[0]['lr0']/decay_factor0
            if self.verbose:
               print ('Reducing learning rate (inner) to %.5f @ T=%s!'%(self.param_groups[0]['lr0'], self.steps))
            
    def update_regularizer(self, decay_factor=None, decay_factor0=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if decay_factor0 != None:
            self.param_groups[0]['lr0'] = self.param_groups[0]['lr0']/decay_factor0
            if self.verbose:
               print ('Reducing learning rate (inner) to %.5f @ T=%s!'%(self.param_groups[0]['lr0'], self.steps))
        if self.verbose:
           print ('Updating regularizer @ T=%s!'%(self.steps))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0
        
        
