import torch
import numpy as np
import torch.nn as nn
from ..utils.utils import check_tensor_shape

class MIDAM_softmax_pooling_loss(nn.Module):
    r"""
    Multiple Instance Deep AUC Maximization with stochastic Smoothed-MaX (MIDAM-smx) Pooling. This loss is used for optimizing the AUROC under Multiple Instance Learning (MIL) setting. 
    The Smoothed-MaX Pooling is defined as

    .. math::
        h(\mathbf w; \mathcal X) = \tau \log\left(\frac{1}{|\mathcal X|}\sum_{\mathbf x\in\mathcal X}\exp(\phi(\mathbf w; \mathbf x)/\tau)\right)

    where :math:`\phi(\mathbf w;\mathbf x)` is the prediction score for instance :math:`\mathbf x` and :math:`\tau>0` is a hyperparameter. 
    We optimize the following AUC loss with the Smoothed-MaX Pooling:

    .. math::
        \min_{\mathbf w\in\mathbb R^d,(a,b)\in\mathbb R^2}\max_{\alpha\in\Omega}F\left(\mathbf w,a,b,\alpha\right)&:= \underbrace{\hat{\mathbb E}_{i\in\mathcal D_+}\left[(h(\mathbf w; \mathcal X_i) - a)^2 \right]}_{F_1(\mathbf w, a)} \\
        &+ \underbrace{\hat{\mathbb E}_{i\in\mathcal D_-}\left[(h(\mathbf w; \mathcal X_i) - b)^2 \right]}_{F_2(\mathbf w, b)} \\
        &+ \underbrace{2\alpha (c+ \hat{\mathbb E}_{i\in\mathcal D_-}h(\mathbf w; \mathcal X_i) - \hat{\mathbb E}_{i\in\mathcal D_+}h(\mathbf w; \mathcal X_i)) - \alpha^2}_{F_3(\mathbf w, \alpha)},

    The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.MIDAM`. The stochastic pooling loss only requires partial data from each bag in the mini-batch For the more details about the formulations, please refer to the original paper [1]_.

    Args:
        data_len (int): number of training samples.
        margin (float, optional): margin parameter for AUC loss (default: ``0.5``).
        tau (float): temperature parameter for smoothed max pooling (default: ``0.1``).
        gamma (float, optional): moving average parameter for pooling operation (default: ``0.9``).
        device (torch.device, optional): the device used for computing loss, e.g., 'cpu' or 'cuda' (default: ``None``)

    Example:
        >>> loss_fn = MIDAM_softmax_pooling_loss(data_len=data_length, margin=margin, tau=tau, gamma=gamma)
        >>> preds = torch.randn(32, 1, requires_grad=True)
        >>> target = torch.empty(32 dtype=torch.long).random_(1)
        >>> # in practice, index should be the indices of your data (bag-index for multiple instance learning).
        >>> loss = loss_fn(exps=preds, y_true=target, index=torch.arange(32)) 
        >>> loss.backward()

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning 2023.
           https://arxiv.org/abs/2305.08040
           
    .. note::
           To use :class:`~libauc.losses.MIDAM_softmax_pooling_loss`, we need to track index for each sample in the training dataset. To do so, see the example below:

           .. code-block:: python

               class SampleDataset (torch.utils.data.Dataset):
                    def __init__(self, inputs, targets):
                        self.inputs = inputs
                        self.targets = targets
                    def __len__ (self) :
                        return len(self.inputs)
                    def __getitem__ (self, index):
                        data = self.inputs[index]
                        target = self.targets[index]
                        return data, target, index

    .. note::
           Practical tips: 
           
           - ``gamma`` is a parameter which is better to be tuned in the range (0, 1) for better performance. Some suggested values are ``{0.1, 0.3, 0.5, 0.7, 0.9}``.
           - ``margin`` can be tuned in as ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
           - ``tau`` can be tuned in the range (0.1, 10) ance. Some suggested values are ``{0.1, 0.3, 0.5, 0.7, 0.9}``.
           - ``margin`` can be tuned in ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
    """
    def __init__(self, data_len, margin=1.0, tau=0.1, gamma=0.9, device=None):
        super(MIDAM_softmax_pooling_loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.gamma = gamma
        self.tau = tau
        self.data_len = data_len
        self.s = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device) 
        self.a = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, requires_grad=False, device=self.device)
        self.margin = margin

    def update_smoothing(self, decay_factor):
        self.gamma = self.gamma/decay_factor

    def forward(self, y_pred, y_true, index): 
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))
        self.s[index] = (1-self.gamma) * self.s[index] + self.gamma * y_pred.detach()
        vs = self.s[index]
        index_p = (y_true == 1)
        index_n = (y_true == 0)
        s_p = vs[index_p]   
        s_n = vs[index_n]   
        logs_p = self.tau*torch.log(s_p)
        logs_n = self.tau*torch.log(s_n)
        gw_ins_p = y_pred[index_p]/s_p 
        gw_ins_n = y_pred[index_n]/s_n
        gw_p = torch.mean(2*self.tau*(logs_p-self.a.detach())*gw_ins_p)
        gw_n = torch.mean(2*self.tau*(logs_n-self.b.detach())*gw_ins_n)
        gw_s = self.alpha.detach()* self.tau * (torch.mean(gw_ins_n) - torch.mean(gw_ins_p))
        ga = torch.mean((logs_p - self.a)**2)
        gb = torch.mean((logs_n - self.b)**2)
        loss = gw_p + gw_n + gw_s + ga + gb
        return loss


class MIDAM_attention_pooling_loss(nn.Module):
    r"""
    Multiple Instance Deep AUC Maximization with stochastic Attention (MIDAM-att) Pooling is used for optimizing the AUROC under Multiple Instance Learning (MIL) setting. 
    The Attention Pooling is defined as

    .. math::
        h(\mathbf w; \mathcal X) = \sigma(\mathbf w_c^{\top}E(\mathbf w; \mathcal X)) = \sigma\left(\sum_{\mathbf x\in\mathcal X}\frac{\exp(g(\mathbf w; \mathbf x))\delta(\mathbf w;\mathbf x)}{\sum_{\mathbf x'\in\mathcal X}\exp(g(\mathbf w; \mathbf x'))}\right),

    where :math:`g(\mathbf w;\mathbf x)` is a parametric function, e.g., :math:`g(\mathbf w; \mathbf x)=\mathbf w_a^{\top}\text{tanh}(V e(\mathbf w_e; \mathbf x))`, where :math:`V\in\mathbb R^{m\times d_o}` and :math:`\mathbf w_a\in\mathbb R^m`. 
    And :math:`\delta(\mathbf w;\mathbf x) = \mathbf w_c^{\top}e(\mathbf w_e; \mathbf x)` is the prediction score from each instance, which will be combined with attention weights.
    We optimize the following AUC loss with the Attention Pooling:

    .. math::
        \min_{\mathbf w\in\mathbb R^d,(a,b)\in\mathbb R^2}\max_{\alpha\in\Omega}F\left(\mathbf w,a,b,\alpha\right)&:= \underbrace{\hat{\mathbb E}_{i\in\mathcal D_+}\left[(h(\mathbf w; \mathcal X_i) - a)^2 \right]}_{F_1(\mathbf w, a)} \\
        &+ \underbrace{\hat{\mathbb E}_{i\in\mathcal D_-}\left[(h(\mathbf w; \mathcal X_i) - b)^2 \right]}_{F_2(\mathbf w, b)} \\
        &+ \underbrace{2\alpha (c+ \hat{\mathbb E}_{i\in\mathcal D_-}h(\mathbf w; \mathcal X_i) - \hat{\mathbb E}_{i\in\mathcal D_+}h(\mathbf w; \mathcal X_i)) - \alpha^2}_{F_3(\mathbf w, \alpha)},

    The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.MIDAM`. The stochastic pooling loss only requires partial data from each bag in the mini-batch. For the more details about the formulations, please refer to the original paper [1]_.

    Args:
        data_len (int): number of training samples.
        margin (float, optional): margin parameter for AUC loss (default: ``0.5``).
        gamma (float, optional): moving average parameter for numerator and denominator on attention calculation (default: ``0.9``).
        device (torch.device, optional): the device used for computing loss, e.g., 'cpu' or 'cuda' (default: ``None``)

    Example:
        >>> loss_fn = MIDAM_attention_pooling_loss(data_len=data_length, margin=margin, tau=tau, gamma=gamma)
        >>> preds = torch.randn(32, 1, requires_grad=True)
        >>> denoms = torch.rand(32, 1, requires_grad=True) + 0.01
        >>> target = torch.empty(32 dtype=torch.long).random_(1)
        >>> # in practice, index should be the indices of your data (bag-index for multiple instance learning).
        >>> # denoms should be the stochastic denominator values output from your model.
        >>> loss = loss_fn(sn=preds, sd=denoms, y_true=target, index=torch.arange(32)) 
        >>> loss.backward()

    Reference:
        .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
           "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
           In International Conference on Machine Learning 2023.
           https://arxiv.org/abs/2305.08040
    
    .. note::
           To use :class:`~libauc.losses.MIDAM_attention_pooling_loss`, we need to track index for each sample in the training dataset. To do so, see the example below:

           .. code-block:: python

               class SampleDataset (torch.utils.data.Dataset):
                    def __init__(self, inputs, targets):
                        self.inputs = inputs
                        self.targets = targets
                    def __len__ (self) :
                        return len(self.inputs)
                    def __getitem__ (self, index):
                        data = self.inputs[index]
                        target = self.targets[index]
                        return data, target, index

    .. note::
           Practical tips: 
           
           - ``gamma`` is a parameter which is better to be tuned in the range (0, 1) for better performance. Some suggested values are ``{0.1, 0.3, 0.5, 0.7, 0.9}``.
           - ``margin`` can be tuned in as ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
    """
    def __init__(self, data_len, margin=1.0, gamma=0.9, device=None):
        super(MIDAM_attention_pooling_loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device 
        self.gamma = gamma
        self.data_len = data_len
        self.sn = torch.tensor([1.0]*data_len).view(-1, 1).to(self.device) 
        self.sd = torch.tensor([1.0]*data_len).view(-1, 1).to(self.device)      
        self.a = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, requires_grad=False, device=self.device)
        self.margin = margin

    def update_smoothing(self, decay_factor):
        self.gamma = self.gamma/decay_factor

    def forward(self, y_pred, y_true, index): 
        sn, sd = y_pred 
        sn = check_tensor_shape(sn, (-1, 1))
        sd = check_tensor_shape(sd, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))
        self.sn[index] = (1-self.gamma) * self.sn[index] + self.gamma * sn.detach()
        self.sd[index] = (1-self.gamma) * self.sd[index] + self.gamma * sd.detach()
        vsn = self.sn[index]
        vsd = torch.clamp(self.sd[index], min=1e-8)
        snd = vsn / vsd
        snd = torch.sigmoid(snd)
        gsnd = snd * (1-snd)
        index_p = (y_true == 1)
        index_n = (y_true == 0)
        snd_p = snd[index_p]
        snd_n = snd[index_n]
        gsnd_p = gsnd[index_p]
        gsnd_n = gsnd[index_n]
        gw_att_p = gsnd_p*(1/vsd[index_p]*sn[index_p] - vsn[index_p]/(vsd[index_p]**2)*sd[index_p]) 
        gw_att_n = gsnd_n*(1/vsd[index_n]*sn[index_n] - vsn[index_n]/(vsd[index_n]**2)*sd[index_n])
        gw_p = torch.mean(2*(snd_p-self.a.detach())*gw_att_p)
        gw_n = torch.mean(2*(snd_n-self.b.detach())*gw_att_n)
        gw_s = self.alpha.detach() * (torch.mean(gw_att_n) - torch.mean(gw_att_p))
        ga = torch.mean((snd_p - self.a)**2)
        gb = torch.mean((snd_n - self.b)**2)
        loss = gw_p + gw_n + gw_s + ga + gb
        return loss

class MIDAMLoss(torch.nn.Module):
    r"""
        A high-level wrapper for :obj:`~MIDAM_softmax_pooling_loss` and :obj:`~MIDAM_attention_pooling_loss`.                 

        Example:
            >>> loss_fn = MIDAMLoss(mode='softmax', data_len=N, margin=para)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32 dtype=torch.long).random_(1)
            >>> # in practice, index should be the indices of your data (bag-index for multiple instance learning).
            >>> loss = loss_fn(exps=preds, y_true=target, index=torch.arange(32)) 
            >>> loss.backward()

            >>> loss_fn = MIDAMLoss(mode='attention', data_len=N, margin=para)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> denoms = torch.rand(32, 1, requires_grad=True) + 0.01
            >>> target = torch.empty(32 dtype=torch.long).random_(1)
            >>> # in practice, index should be the indices of your data (bag-index for multiple instance learning).
            >>> # denoms should be the stochastic denominator values output from your model.
            >>> loss = loss_fn(sn=preds, sd=denoms, y_true=target, index=torch.arange(32)) 
            >>> loss.backward()

        Reference:
            .. [1] Dixian Zhu, Bokun Wang, Zhi Chen, Yaxing Wang, Milan Sonka, Xiaodong Wu, Tianbao Yang
               "Provable Multi-instance Deep AUC Maximization with Stochastic Pooling."
               In International Conference on Machine Learning 2023.
               https://arxiv.org/abs/2305.08040
    """
    def __init__(self, mode='attention', **kwargs):
        super(MIDAMLoss, self).__init__()
        assert mode in ['attention', 'softmax'], 'keywords are not found!'  
        self.mode = mode 
        self.loss_fn = self.get_loss(mode, **kwargs)
        self.a = self.loss_fn.a
        self.b = self.loss_fn.b
        self.alpha = self.loss_fn.alpha
        self.margin = self.loss_fn.margin
                                       
    def get_loss(self, mode='attention', **kwargs):
        if mode == 'attention':
           loss = MIDAM_attention_pooling_loss(**kwargs)
        elif mode == 'softmax':
           loss = MIDAM_softmax_pooling_loss(**kwargs)
        else:
            raise ValueError('Out of options!')
        return loss

    def update_smoothing(self, decay_factor):
        self.loss_fn.gamma = self.loss_fn.gamma/decay_factor

    def forward(self, **kwargs):
        return self.loss_fn(**kwargs)
    