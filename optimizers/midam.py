import math
import torch
from torch.optim.optimizer import Optimizer, required 

class MIDAM(torch.optim.Optimizer):
    r"""
        MIDAM (Multiple Instance Deep AUC Maximization) is used for optimizing the :obj:`~libauc.losses.MIDAMLoss` (softmax or attention pooling based AUC loss).
                     
        Notice that :math:`h(\mathbf w; \mathcal X_i)=f_2(f_1 (\mathbf w;\mathcal X_i))` is the bag-level prediction after the pooling operation. Denote that the moving average estimation for bag-level prediction for i-th bag at t-th iteration as :math:`s_i^t`. The gradients estimation are:
            
            .. math::
                G^t_{1,\mathbf w} = \hat{\mathbb E}_{i\in\mathcal S_+^t}\nabla f_1(\mathbf w^t; \mathcal B_{i}^t) \nabla f_2(s^{t-1}_i)\nabla_1 f( f_2(s^{t-1}_i), a^t),
                
            .. math::
                G^t_{2,\mathbf w} = \hat{\mathbb E}_{i\in\mathcal S_-^t}\nabla f_1(\mathbf w^t; \mathcal B_{i}^t) \nabla f_2(s^{t-1}_i)\nabla_1 f( f_2(s^{t-1}_i), b^t),
                
            .. math::
                G^t_{3,\mathbf w} = \alpha^t \cdot\left(\hat{\mathbb E}_{i\in\mathcal S_-^t}\nabla f_1(\mathbf w^t; \mathcal B_{i}^t) \nabla f_2(s^{t-1}_i)\right.  \left.- \hat{\mathbb E}_{i\in\mathcal S_+^t}\nabla f_1(\mathbf w^t; \mathcal B_{i}^t) \nabla f_2(s^{t-1}_i)\right),
                
            .. math::
                G^t_{1,a}  = \hat{\mathbb E}_{i\in\mathcal S_+^t} \nabla_2 f( f_2(s^{t-1}_i), a^t),
                
            .. math::
                G^t_{2, b} =\hat{\mathbb E}_{i\in\mathcal S_-^t} \nabla_2 f( f_2(s^{t-1}_i), b^t),
                
            .. math::
                G^t_{3,\alpha}   = c+ \hat{\mathbb E}_{i\in\mathcal S_-^t}f_2(s^{t-1}_i) - \hat{\mathbb E}_{i\in\mathcal S_+^t}f_2(s^{t-1}_i),
        
        The key update steps for the stochastic optimization are summarized as follows:

            1. Initialize :math:`\mathbf s^0=0, \mathbf v^0=\mathbf 0, a=0, b=0, \mathbf w`
            2. For :math:`t=1, \ldots, T`:
            3. :math:`\hspace{5mm}` Sample a batch of positive bags :math:`\mathcal S_+^t\subset\mathcal D_+` and a batch of negative bags :math:`\mathcal S_-^t\subset\mathcal D_-`. 
            4. :math:`\hspace{5mm}` For each :math:`i \in \mathcal S^t=\mathcal S_+^t\cup \mathcal S_-^t`:
            5. :math:`\hspace{5mm}` Sample a mini-batch of instances :math:`\mathcal B^t_i\subset\mathcal X_i` and update:

                .. math::

                    s^t_i =  (1-\gamma_0)s^{t-1}_i  + \gamma_0 f_1(\mathbf w^t; \mathcal B_{i}^t)
                    
            6. :math:`\hspace{5mm}` Update stochastic gradient estimator of :math:`(\mathbf w, a, b)`: 
            
                .. math::
                    \mathbf v_1^t =\beta_1\mathbf v_1^{t-1} + (1-\beta_1)(G^t_{1,\mathbf w} + G^t_{2,\mathbf w} + G^t_{3,\mathbf w})
                
                .. math::
                    \mathbf v_2^t =\beta_1\mathbf v_2^{t-1} + (1-\beta_1)G^t_{1,a}
                    
                .. math::
                    \mathbf v_3^t =\beta_1\mathbf v_3^{t-1} + (1-\beta_1)G^t_{2,b}

            6. :math:`\hspace{5mm}` Update :math:`(\mathbf w^{t+1}, a^{t+1}, b^{t+1}) = (\mathbf w^t, a^t, b^t) - \eta \mathbf v^t` (or Adam style)
            7. :math:`\hspace{5mm}` Update :math:`\alpha^{t+1}  = \Pi_{\Omega}[\alpha^t +  \eta' (G^t_{3,\alpha} - \alpha^t)]`         
            
            For more details, please refer to the paper `Provable Multi-instance Deep AUC Maximization with Stochastic Pooling.`

        Args:
            params (iterable): iterable of parameters to optimize
            loss_fn (callable): loss function used for optimization (default: ``None``)
            lr (float): learning rate (default: ``0.1``)
            momentum (float, optional): momentum factor for 'sgd' mode (default: ``0.1``)
            weight_decay (float, optional): weight decay (L2 penalty) (default: ``1e-5``)
            device (torch.device, optional): the device used for optimization, e.g., 'cpu' or 'cuda' (default: ``None``)
            
        Example:
            >>> optimizer = libauc.optimizers.MIDAM(params=model.parameters(), loss_fn=loss_fn, lr=0.1, momentum=0.1)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()


        Reference:
            .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
               "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
               In International Conference on Machine Learning, pp. xxxxx-xxxxx. PMLR, 2023.
               https://prepare-arxiv?
    """

    def __init__(self, params, loss_fn, lr=required, momentum=0, weight_decay=0, device=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
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

        self.params = list(params) 
        if self.a is not None and self.b is not None:
           self.params = self.params + [self.a, self.b]
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.T = 0
        defaults = dict(lr=lr, momentum=momentum, margin=self.margin, a=self.a, b=self.b, alpha=self.alpha, weight_decay=weight_decay)
        super(MIDAM, self).__init__(self.params, defaults)
        

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    

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
            momentum = group['momentum']
            self.lr = group['lr']
            alpha = group['alpha']
            m = group['margin'] 
            a = group['a']
            b = group['b']
            
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay) 
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - momentum) 
                    d_p = buf

                p.add_(d_p, alpha=-group['lr'])
                
            if alpha is not None: 
              alpha.data = torch.clip(alpha.data + group['lr']*((m + b.data - a.data)-alpha.data), min=0.0)
        self.T = self.T + 1
        return loss
    
    def update_lr(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            print('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.T))
        print('Updating regularizer @ T=%s!'%(self.T))       

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
