import warnings
import torch 
import torch.nn.functional as F
from .surrogate import get_surrogate_loss
from ..utils.utils import check_tensor_shape

__all__ = ['AUCMLoss',
           'CompositionalAUCLoss',
           'AveragePrecisionLoss',
           'APLoss',
           'pAUC_CVaR_Loss',
           'pAUC_DRO_Loss',
           'tpAUC_KL_Loss',
           'pAUCLoss',
           'PairwiseAUCLoss',
           'meanAveragePrecisionLoss',
           'MultiLabelAUCMLoss',
           'MultiLabelpAUCLoss', 
           'mAUCMLoss', 
           'mAPLoss',
           'mPAUCLoss',
           ]


class AUCMLoss(torch.nn.Module):
    r"""
        AUC-Margin loss with squared-hinge surrogate loss for optimizing AUROC. The objective function is defined as:

        .. math::

            \min _{\substack{\mathbf{w} \in \mathbb{R}^d \\(a, b) \in \mathbb{R}^2}} \max _{\alpha \in \mathbb{R^+}} f(\mathbf{w}, a, b, \alpha):=\mathbb{E}_{\mathbf{z}}[F(\mathbf{w}, a, b, \alpha ; \mathbf{z})]

        where 
        
        .. math::

            F(\mathbf{w},a,b,\alpha; \mathbf{z}) &=(1-p)(h_{\mathbf{w}}(x)-a)^2\mathbb{I}_{[y=1]} +p(h_{\mathbf{w}}(x)-b)^2\mathbb{I}_{[y=-1]} \\
            &+2\alpha(p(1-p)m+ p h_{\mathbf{w}}(x)\mathbb{I}_{[y=-1]}-(1-p)h_{\mathbf{w}}(x)\mathbb{I}_{[y=1]})\\
            &-p(1-p)\alpha^2

        :math:`h_{\mathbf{w}}` is the prediction scoring function, e.g., deep neural network, :math:`p` is the ratio of positive samples to all samples, :math:`a`, :math:`b` are the running statistics of 
        the positive and negative predictions, :math:`\alpha` is the auxiliary variable derived from the problem formulation and :math:`m` is the margin term. We denote this version of AUCMLoss as ``v1``.
        
        To remove the class prior :math:`p` in the above formulation, we can write the new objective function as follow:

         .. math::

            f(\mathbf{w},a,b,\alpha) &= \mathbb{E}_{y=1}[(h_{\mathbf{w}}(x)-a)^2] + \mathbb{E}_{y=-1}[(h_{\mathbf{w}}(x)-b)^2] \\
            &+2\alpha(m + \mathbb{E}_{y=-1}[h_{\mathbf{w}}(x)] - \mathbb{E}_{y=1}[h_{\mathbf{w}}(x)])\\
            &-\alpha^2

        We denote this version of AUCMLoss as ``v2``. The optimization algorithm for solving the above objectives are implemented as :obj:`~libauc.optimizers.PESG`. For the derivations, please refer to the original paper [1]_.

        args:
            margin (float): margin for squared-hinge surrogate loss (default: ``1.0``).
            imratio (float, optional): the ratio of the number of positive samples to the number of total samples in the training dataset. 
                                       If this value is not given, the mini-batch statistics will be used instead.
            version (str, optional): whether to include prior :math:`p` in the objective function (default: ``'v1'``).
           

        Example:
            >>> loss_fn = libauc.losses.AUCMLoss(margin=1.0)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> loss = loss_fn(preds, target)
            >>> loss.backward()

        .. note::
            To use ``v2`` of AUCMLoss, plesae set ``version='v2'``. Otherwise, the default version is ``v1``. The ``v2`` version requires the use of :obj:`~libauc.sampler.DualSampler`.

        .. note::
            Practial Tips: 

            - ``epoch_decay`` is a regularization parameter similar to `weight_decay` that can be tuned in the same range.
            - For complex tasks, it is recommended to use regular loss to pretrain the model, and then switch to AUCMLoss for finetuning with a smaller learning rate. 

        Reference:
            .. [1] Yuan, Zhuoning, Yan, Yan, Sonka, Milan, and Yang, Tianbao.
               "Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification."
               Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
               https://arxiv.org/abs/2012.03173
    """
    def __init__(self, 
                 margin=1.0, 
                 imratio=None,
                 version='v1',
                 device=None):
        super(AUCMLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p = imratio
        self.version = version
        assert version in ['v1', 'v2'], "Input value is not valid! Possible values are ['v1', 'v2']."
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True)

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true, auto=True, **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        pos_mask = (1==y_true).float()
        neg_mask = (0==y_true).float()

        if sum(pos_mask) == 0:
            warnings.warn("Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!", UserWarning)

        if self.version == 'v1': 
            if auto or self.p == None:
                self.p = pos_mask.sum()/y_true.shape[0]
            loss = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float())   + \
                    2*self.alpha*(self.p*(1-self.p)*self.margin + \
                    torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                    self.p*(1-self.p)*self.alpha**2
        else:
            loss = self.mean((y_pred - self.a)**2*pos_mask) + \
                   self.mean((y_pred - self.b)**2*neg_mask) + \
                        2*self.alpha*(self.margin + \
                        self.mean((y_pred*neg_mask) - self.mean(y_pred*pos_mask)) )- \
                        self.alpha**2
        return loss

class CompositionalAUCLoss(torch.nn.Module):
    r"""
        Compositional AUC loss with squared-hinge surrogate loss for optimizing AUROC. The objective is defined as 

        .. math::

            L_{\mathrm{AUC}}\left(\mathbf{w}-\alpha \nabla L_{\mathrm{CE}}(\mathbf{w})\right)

        where :math:`L_{\mathrm{AUC}}` refers to :obj:`~AUCMLoss`, :math:`L_{\mathrm{CE}}` refers to :obj:`~CrossEntropyLoss` and math:`\alpha` refer to the step size for inner updates. 

        The optimization algorithm for solving this objective is implemented as :obj:`~libauc.optimizers.PDSCA`. For the derivations, please refer to the original paper [2]_.

        args:
            margin (float): margin for squared-hinge surrogate loss (default: ``1.0``).
            imratio (float, optional): the ratio of the number of positive samples to the number of total samples in the training dataset. If this value is not given, the mini-batch statistics will be used instead.
            k (int, optional): number of steps for inner updates. For example, when k is set to 2, the optimizer will alternately execute two steps optimizing :obj:`~libauc.losses.losses.CrossEntropyLoss` followed by a single step optimizing :obj:`~libauc.losses.auc.AUCMLoss` during training (default: ``1``).
            version (str, optional): whether to include prior :math:`p` in the objective function (default: ``'v1'``).
     
        Example:
            >>> loss_fn = libauc.losses.CompositionalAUCLoss(margin=1.0, k=1)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> loss = loss_fn(preds, target)
            >>> loss.backward()    

        .. note::
            As CompositionalAUCLoss is built on AUCMLoss, there are also two versions of CompositionalAUCLoss. To use ``v2`` version, plesae set ``version='v2'``. Otherwise, the default version is ``v1``. 

        .. note::

            Practial Tips: 

            - By default, ``k`` is set to 1. You may consider increasing it to a larger number to potentially improve performance. 


        Reference:
                .. [2] Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang.
                   "Compositional Training for End-to-End Deep AUC Maximization."
                   International Conference on Learning Representations 2022.
                   https://openreview.net/forum?id=gPvB4pdu_Z
    """
    def __init__(self,  
                 margin=1.0, 
                 k=1, 
                 version='v1',
                 imratio=None, 
                 backend='ce', 
                 l_avg=None,   # todo: loss placeholder
                 l_imb=None,   # todo: loss placeholder
                 device=None):
        super(CompositionalAUCLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        assert k >=1 , 'k >=1!'
        self.margin = margin
        self.p = imratio
        self.version = version
        assert version in ['v1', 'v2'], "Input value is not valid! Possible values are ['v1', 'v2']."
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.L_AVG = F.binary_cross_entropy 
        self.backend = backend  
        self.k = k
        self.step = 0
        
 
    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true, k=None, auto=True, **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        if k !=None: # override k
           self.k = k 

        if self.step % (2*self.k) < self.k:
           self.step += 1
           return self.L_AVG(y_pred, y_true)
        else:
            if self.version == 'v1':
                if auto == True or self.p == None:
                    self.p = (y_true==1).sum()/y_true.shape[0] 
                self.L_AUC = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                             self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float())     + \
                             2*self.alpha*(self.p*(1-self.p)*self.margin + \
                             torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                             self.p*(1-self.p)*self.alpha**2
            
            else:
                pos_mask = (1==y_true).float()
                neg_mask = (0==y_true).float()
                self.L_AUC = self.mean((y_pred - self.a)**2*pos_mask) + \
                             self.mean((y_pred - self.b)**2*neg_mask) + \
                             2*self.alpha*(self.margin + \
                             self.mean((y_pred*neg_mask) - self.mean(y_pred*pos_mask)) ) - \
                             self.alpha**2
            self.step += 1
            return self.L_AUC 


  
class AveragePrecisionLoss(torch.nn.Module):
    r"""
        Average Precision loss with squared-hinge surrogate loss for optimizing AUPRC. The objective is defined as  
        
        .. math::
        
            \min_{\mathbf{w}} P(\mathbf{w})=\frac{1}{n_{+}}\sum\limits_{y_i=1}\frac{-\sum\limits_{s=1}^n\mathbb{I}(y_s=1)\ell(\mathbf{w};\mathbf{x}_s;\mathbf{x}_i)}{\sum\limits_{s=1}^n \ell(\mathbf{w};\mathbf{x}_s;\mathbf{x}_i)}         
       
        where :math:`\ell(\mathbf{w}; \mathbf{x}_s, \mathbf{x}_i)` is a surrogate function of the non-continuous indicator function :math:`\mathbb{I}(h(\mathbf{x}_s)\geq h(\mathbf{x}_i))`, :math:`h(\cdot)` is the prediction function, 
        e.g., deep neural network. 

        The optimization algorithm for solving this objective is implemented as :obj:`~libauc.optimizers.SOAP`. For the derivations, please refer to the original paper [3]_.
        
        This class is also aliased as :obj:`~libauc.losses.auc.APLoss`.

        args:
          data_len (int):  total number of samples in the training dataset.
          gamma (float, optional): parameter for moving average estimator (default: ``0.9``).
          surr_loss (str, optional): the choice for surrogate loss used for problem formulation (default: ``'squared_hinge'``).
          margin (float, optional): margin for squred hinge surrogate loss (default: ``1.0``).

        Example:
            >>> loss_fn = libauc.losses.APLoss(data_len=data_length)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(preds, target, index)
            >>> loss.backward()
            
        .. note::
           To use :class:`~libauc.losses.APLoss`, we need to track index for each sample in the training dataset. To do so, see the example below:

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
            - To further improve the performance, try tuning ``margin`` in the range (0, 1].  Some suggested values: ``{0.6, 0.8, 1.0}``.


        Reference:
            .. [3] Qi, Qi, Youzhi Luo, Zhao Xu, Shuiwang Ji, and Tianbao Yang. 
                "Stochastic optimization of areas under precision-recall curves with provable convergence." 
                In Advances in Neural Information Processing Systems 34 (2021): 1752-1765.
                https://proceedings.neurips.cc/paper/2021/file/0dd1bc593a91620daecf7723d2235624-Paper.pdf
    """
    def __init__(self, 
                 data_len, 
                 gamma=0.9, 
                 margin=1.0,  
                 surr_loss='squared_hinge', 
                 device=None):
        super(AveragePrecisionLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device           
        self.u_all = torch.tensor([0.0]*data_len).reshape(-1, 1).to(self.device).detach()
        self.u_pos = torch.tensor([0.0]*data_len).reshape(-1, 1).to(self.device).detach()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = get_surrogate_loss(surr_loss)
        
    def forward(self, y_pred, y_true, index, **kwargs): 
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))
        pos_mask = (y_true == 1).squeeze()
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask] # indices for positive samples only     
        f_ps  = y_pred[pos_mask]    # shape: (len(f_ps), 1)
        f_all = y_pred.squeeze()    # shape: (len(f_all), )
        surr_loss = self.surrogate_loss(self.margin, (f_ps - f_all)) # shape: (len(f_ps), len(f_all))
        pos_surr_loss = surr_loss * pos_mask  
        self.u_all[index] = (1 - self.gamma) * self.u_all[index] + self.gamma * (surr_loss.mean(1, keepdim=True)).detach()
        self.u_pos[index] = (1 - self.gamma) * self.u_pos[index] + self.gamma * (pos_surr_loss.mean(1, keepdim=True)).detach()
        p = (self.u_pos[index] - (self.u_all[index]) * pos_mask) / (self.u_all[index] ** 2)   # shape of p: len(f_ps)*len(y_pred)
        p.detach_()
        loss = torch.mean(p * surr_loss)
        return loss


class pAUC_CVaR_Loss(torch.nn.Module):
    r"""
        Partial AUC loss based on DRO-CVaR to optimize One-way Partial AUROC (OPAUC). The loss focuses on optimizing OPAUC in the range [0, beta] for false positive rate. The objective is defined as

        .. math::

            F(\mathbf w, \mathbf s) = \frac{1}{n_+}\sum_{\mathbf x_i\in\mathcal S_+} \left(s_i  +  \frac{1}{\beta n_-}\sum_{\mathbf x_j\in \mathcal S_-}(L(\mathbf w; \mathbf x_i, \mathbf x_j) - s_i)_+\right)

        where :math:`L(\mathbf w; \mathbf x_i, \mathbf x_j)` is the surrogate pairwise loss function for one positive data and one negative data, e.g., squared hinge loss, logitstic loss, etc. :math:`\mathbf s` is the dual variable from DRO-CVaR formulation that is minimized in the loss function. For a positive data :math:`\mathbf x_i`, any pairwise losses samller than :math:`s_i` are truncated. Therefore, the loss function focus on the harder negative data; as a consequence, the `pAUC_CVaR_Loss` optimize the upper bounded FPR (false positive rate) of pAUC region.
        
        This loss optimizes OPAUC in the range [0, beta] for False Positive Rate (FPR). The optimization algorithm for solving this objective is implemented as :obj:`~libauc.optimizers.SOPA`. For the derivations, please refer to the original paper [4]_.
        
        Args:
            data_len (int):  total number of samples in the training dataset.
            pos_len (int): total number of positive samples in the training dataset.
            margin (float, optional): margin term for squared-hinge surrogate loss (default: ``1.0``).
            beta (float): upper bound of False Positive Rate (FPR) used for optimizing pAUC (default: ``0.2``).
            eta (float): stepsize for update the dual variables for DRO-CVaR formulation (default: ``0.1``).
            surr_loss (string, optional): surrogate loss used in the problem formulation (default: ``'squared_hinge'``).

        Example:
            >>> loss_fn = pAUC_CVaR_loss(data_len=data_length, pos_len=pos_length)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(preds, target, index) 
            >>> loss.backward()


        .. note::
           To use :class:`~libauc.losses.pAUC_CVaR_Loss`, we need to track index for each sample in the training dataset. To do so, see the example below:

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

            - ``margin`` can be tuned in ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
            - ``beta`` can be tuned in the range (0.1, 0.9), ideally based on task requirement for FPR.
            - ``eta`` can be tuned in ``{10, 1.0, 0.1, 0.01}`` for better performance. 


        Reference:
            .. [4] Zhu, Dixian and Li, Gang and Wang, Bokun and Wu, Xiaodong and Yang, Tianbao.
               "When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee."
               In International Conference on Machine Learning, pp. 27548-27573. PMLR, 2022.
               https://proceedings.mlr.press/v162/zhu22g.html
    """                                   
    def __init__(self, 
                 data_len, 
                 pos_len,
                 num_neg=None, 
                 margin=1.0, 
                 beta=0.2, 
                 eta=0.1,
                 surr_loss='squared_hinge', 
                 device=None):
        super(pAUC_CVaR_Loss, self).__init__()    
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin    
        self.beta = beta                  
        self.eta = eta
        self.num_neg = num_neg
        self.data_len = data_len
        self.pos_len = pos_len
        self.u_pos = torch.tensor([0.0]*data_len).reshape(-1, 1).to(self.device).detach()               
        self.surrogate_loss = get_surrogate_loss(surr_loss)
                                       

    def forward(self, y_pred, y_true, index, auto=True, **kwargs):       
        if auto:
           self.num_neg = (y_true == 0).float().sum()
           assert self.num_neg > 0, 'There is no negative sample in the data!'
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))
        pos_mask = (y_true == 1).squeeze() 
        neg_mask = (y_true == 0).squeeze() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred):    
            index = index[pos_mask]        # indices for positive samples       
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1)   
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns))  
        surr_loss = self.surrogate_loss(self.margin, f_ps - f_ns)   # shape: (len(f_ps), len(f_ns)) 
        p = surr_loss > self.u_pos[index]
        self.u_pos[index] = self.u_pos[index]-self.eta/self.pos_len*(1 - p.sum(dim=1, keepdim=True)/(self.beta*self.num_neg))
        p.detach_()
        loss = torch.mean(p * surr_loss) / self.beta
        return loss
                                       
                                     
class pAUC_DRO_Loss(torch.nn.Module):
    r"""
        Partial AUC loss based on KL-DRO to optimize One-way Partial AUROC (OPAUC). In contrast to conventional AUC, partial AUC pays more attention to partial difficult samples. By leveraging the Distributionally Robust Optimization (DRO), the objective is defined as

            .. math::
               \min_{\mathbf{w}}\frac{1}{n_+}\sum_{\mathbf{x}_i\in\mathbf{S}_+} \max_{\mathbf{p}\in\Delta} \sum_j p_j L(\mathbf{w}; \mathbf{x}_i, \mathbf{x}_j) - \lambda \text{KL}(\mathbf{p}, 1/n)

        Then the objective is reformulated as follows to develop an algorithm.

            .. math::
               \min_{\mathbf{w}}\frac{1}{n_+}\sum_{\mathbf{x}_i \in \mathbf{S}_+}\lambda \log \frac{1}{n_-}\sum_{\mathbf{x}_j \in \mathbf{S}_-}\exp\left(\frac{L(\mathbf{w}; \mathbf{x}_i, \mathbf{x}_j)}{\lambda}\right)

        where :math:`L(\mathbf{w}; \mathbf{x_i}, \mathbf{x_j})` is the surrogate pairwise loss function for one positive data and one negative data, e.g., squared hinge loss, :math:`\mathbf{S}_+` and :math:`\mathbf{S}_-` denote the subsets of the dataset which contain only positive samples and negative samples, respectively.

        The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.SOAPs`. For the derivation of the above formulation, please refer to the original paper [4]_.


        Args:
            data_len (int):  total number of samples in the training dataset.
            gamma (float): parameter for moving average estimator (default: ``0.9``).
            surr_loss (string, optional): surrogate loss used in the problem formulation (default: ``'squared_hinge'``).
            margin (float, optional): margin for squared-hinge surrogate loss (default: ``1.0``).
            Lambda (float, optional): weight for KL divergence regularization, e.g., 0.1, 1.0, 10.0 (default: ``1.0``).

        Example:
            >>> loss_fn = libauc.losses.pAUC_DRO_Loss(data_len=data_length, gamma=0.9, Lambda=1.0)
            >>> preds  = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(preds, target, index)
            >>> loss.backward()

        .. note::
           To use :class:`~libauc.losses.pAUC_DRO_Loss`, we need to track index for each sample in the training dataset. To do so, see the example below:

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
            - ``margin`` can be tuned in ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
            - ``Lambda`` can be tuned in the range (0.1, 10) for better performance. 

    """                        
    def __init__(self, 
                 data_len, 
                 gamma=0.9,
                 margin=1.0,
                 Lambda=1.0, 
                 surr_loss='squared_hinge', 
                 device=None):
        super(pAUC_DRO_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device     
        self.data_len = data_len      
        self.u_pos = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma 
        self.Lambda = Lambda                           
        self.surrogate_loss = get_surrogate_loss(surr_loss)
    
    def forward(self, y_pred, y_true, index, **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze() 
        neg_mask = (y_true == 0).squeeze() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only       
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.surrogate_loss(self.margin, (f_ps - f_ns))  # shape: (len(f_ps), len(f_ns))                       
        exp_loss = torch.exp(surr_loss/self.Lambda)
        self.u_pos[index] = (1 - self.gamma) * self.u_pos[index] + self.gamma * (exp_loss.mean(1, keepdim=True).detach())
        p = exp_loss/self.u_pos[index]    # shape: (len(f_ps), len(f_ns))                       
        p.detach_()
        loss = torch.mean(p * surr_loss)
        return loss
                                      
                                       
class tpAUC_KL_Loss(torch.nn.Module):
    r"""
        Partial AUC loss based on DRO-KL to optimize two-way partial AUROC. The objective function is defined as
        
        .. math::
        
            F(\mathbf w; \phi_{kl}, \phi_{kl})=  \lambda'\log \mathrm E_{\mathbf x_i\sim\mathcal S_+}\left(\mathrm E_{\mathbf x_j\sim\mathcal S_-}\exp(\frac{L(\mathbf w; \mathbf x_i,\mathbf x_j)}{\lambda})\right)^{\frac{\lambda}{\lambda'}}
        where :math:`L(\mathbf w; \mathbf x_i, \mathbf x_j)` is the surrogate pairwise loss function for one positive data and one negative data, e.g., squared hinge loss, logitstic loss, etc. In this formulation, we implicitly handle the :math:`\alpha` and :math:`\beta` range of TPAUC by tuning :math:`\lambda` and :math:`\lambda'` (we rename :math:`\lambda` as Lambda and :math:`\lambda'` as tau for coding purpose). The loss focuses on both harder positive and harder negative samples, hence can optimize the TPAUC on the left corner space of the AUROC curve.
        
        The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.SOTAs`. For the derivation of the above formulation, please refer to the original paper [4]_.
        
        Args:
            data_len (int):  total number of samples in the training dataset.
            margin (float, optional): margin term used in surrogate loss (default: ``1.0``).
            Lambda (float, optional): KL regularization for negative samples (default: ``1.0``).
            tau (float, optional): KL regularization for positive samples (default: ``1.0``).
            gammas (Tuple[float, float], optional): coefficients used for moving average estimation for composite functions. (default: ``(0.9, 0.9)``)
            surr_loss (string, optional): surrogate loss used in the problem formulation (default: ``'squared_hinge'``).
       
        Example:
            >>> loss_fn = tpAUC_KL_Loss(data_len=data_length)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(preds, target, index)
            >>> loss.backward()

        .. note::
           To use :class:`~libauc.losses.tpAUC_KL_Loss`, we need to track index for each sample in the training dataset. To do so, see the example below:

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

            - ``gammas`` are parameters which are better to be tuned in the range (0, 1) for better performance. Some suggested values are ``{(0.1, 0.1), (0.5,0.5), (0.9,0.9)}``.
            - ``margin`` can be tuned in ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
            - ``Lambda`` and ``tau`` can be tuned in the range (0.1, 10) for better performance.

    """                                
    def __init__(self, 
                 data_len, 
                 tau=1.0, 
                 Lambda=1.0, 
                 gammas=(0.9, 0.9),
                 margin=1.0, 
                 surr_loss='squared_hinge', 
                 device=None):
        super(tpAUC_KL_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device     
        self.gamma0 = gammas[0]
        self.gamma1 = gammas[1]
        self.Lambda = Lambda
        self.tau = tau
        self.data_len = data_len
        self.u_pos = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.w = 0.0 
        self.margin = margin                           
        self.surrogate_loss = get_surrogate_loss(surr_loss)
                                       
    def forward(self, y_pred, y_true, index, **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))
        pos_mask = (y_true == 1).squeeze() 
        neg_mask = (y_true == 0).squeeze() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) ==len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only 
        f_ps = y_pred[pos_mask]       # shape: (len(f_ps), 1)   
        f_ns = y_pred[neg_mask].squeeze() # shape: (len(f_ns), 1)   
        surr_loss = self.surrogate_loss(self.margin, f_ps - f_ns) # shape: (len(f_ps), len(f_ns))  
        exp_loss = torch.exp(surr_loss/self.Lambda).detach()
        self.u_pos[index] = (1 - self.gamma0) * self.u_pos[index] + self.gamma0 * (exp_loss.mean(1, keepdim=True))
        self.w = (1 - self.gamma1) * self.w + self.gamma1 * (torch.pow(self.u_pos[index], self.Lambda/self.tau).mean())
        p = torch.pow(self.u_pos[index], self.Lambda/self.tau - 1) * exp_loss/self.w                     
        p.detach_()
        loss = torch.mean(p * surr_loss)
        return loss
                           
class pAUCLoss(torch.nn.Module):
    r"""
        A wrapper for Partial AUC losses to optimize One-way and Two-way Partial AUROC. By default, One-way Partial AUC (OPAUC) refers to :obj:`~SOPAs` and  
        Two-way Partial AUC (TPAUC) refers to :obj:`~SOTAs`. The usage for each loss is same as the original loss. 

        args:
            mode (str): the specific loss function to be used in the backend (default: '1w').
            **kwargs: the required arguments for the selected loss function. 

        Example:
            >>> loss_fn = pAUCLoss(mode='1w', data_len=data_length)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(preds, target, index)
            >>> loss.backward() 
    """
    def __init__(self, mode='1w', **kwargs):
        super(pAUCLoss, self).__init__()
        assert mode in ['SOPA', 'SOPAs', 'SOTAs', '1w', '2w'], 'Loss is not implemented!'  
        self.mode = mode 
        self.loss_fn = self.get_loss(mode, **kwargs)
                                       
    def get_loss(self, mode='1w', **kwargs):
        if mode == 'SOPA':
           loss = pAUC_CVaR_Loss(**kwargs)
        elif mode == 'SOPAs' or mode=='1w':
           loss = pAUC_DRO_Loss(**kwargs)
        elif mode == 'SOTAs' or mode=='2w':
           loss = tpAUC_KL_Loss(**kwargs)
        else:
            raise ValueError('Out of options!')
        return loss
   
    def forward(self, y_pred, y_true, index, **kwargs):
        return self.loss_fn(y_pred, y_true, index, **kwargs)
    
    
class PairwiseAUCLoss(torch.nn.Module): 
    r"""
        Pairwise AUC loss to optimize AUROC based on different surrogate losses. For optimizing this objective, we can use existing optimizers in LibAUC or PyTorch such as, :obj:`~libauc.optimizers.SGD`, :obj:`~libauc.optimizers.Adam, :obj:`~libauc.optimizers.AdamW`.

        args:
            surr_loss (str): surrogate loss for optimizing pairwise AUC loss. The available options are 'logistic', 
                'squared', 'squared_hinge', 'barrier_hinge' (default: ``'squared_hinge'``).
            hparam (float or tuple, optional): abstract hyper parameter for different surrogate loss. In particular, the available options are:

                - :obj:`squared` with tunable margin term (default: ``1.0``).
                - :obj:`squared_hinge` with tunable margin term (default: ``1.0``).
                - :obj:`logistic` with tunable scaling term (default: ``1.0``).
                - :obj:`barrier_hinge` with tunable a tuple of (scale, margin) (default: ``(1.0, 1.0)``).

        Example:
            >>> loss_fn = PairwiseAUCLoss(surr_loss='squared', hparam=0.5)
            >>> y_pred = torch.randn(32, requires_grad=True)
            >>> y_true = torch.empty(32, dtype=torch.long).random_(2)
            >>> loss = loss_fn(y_pred, y_true)
            >>> loss.backward()
    """
    def __init__(self, surr_loss='logistic', hparam=1.0): 
        super(PairwiseAUCLoss, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparam = hparam                   
        self.surrogate_loss = get_surrogate_loss(surr_loss)

    def forward(self, y_pred, y_true, index=None, **kwargs): 
        y_pred = check_tensor_shape(y_pred, (-1,))
        y_true = check_tensor_shape(y_true, (-1,))
        pos_mask = (y_true == 1).squeeze() 
        neg_mask = (y_true == 0).squeeze() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        f_ps = y_pred[pos_mask]
        f_ns = y_pred[neg_mask].squeeze()
        loss = self.surrogate_loss(self.hparam, f_ps - f_ns)  
        return loss.mean()


class MultiLabelAUCMLoss(torch.nn.Module):
    r"""
        AUC-Margin loss with squared-hinge surrogate loss to optimize multi-label AUROC. This is an extension of :obj:`~libauc.losses.AUCMLoss`. 

        Args:
            margin (float): margin term for squared-hinge surrogate loss. (default: ``1.0``)
            num_labels (int): number of labels for the dataset.
            imratio (float, optional): the ratio of the number of positive samples to the number of total samples in the training dataset. 
                                        If this value is not given, the mini-batch statistics will be used instead.
            version (str, optional): whether to include prior :math:`p` in the objective function (default: ``'v1'``).
            
        This class is also aliased as :obj:`~libauc.losses.auc.mAUCMLoss`.

        Example:
            >>> loss_fn = MultiLabelAUCMLoss(margin=1.0, num_labels=10)
            >>> y_pred = torch.randn(32, 10, requires_grad=True)
            >>> y_true = torch.empty(32, dtype=torch.long).random_(2)
            >>> loss = loss_fn(y_pred, y_true)
            >>> loss.backward()

        Reference:
            .. [5] Zhuoning Yuan, Dixian Zhu, Zi-Hao Qiu, Gang Li, Xuanhui Wang, Tianbao Yang.
                   "LibAUC: A Deep Learning Library for X-Risk Optimization."
                   29th SIGKDD Conference on Knowledge Discovery and Data Mining.
                   https://arxiv.org/abs/2306.03065
  
    """
    def __init__(self, 
                 margin=1.0,
                 version='v1', 
                 imratio=None, 
                 num_labels=10, 
                 device=None):
        super(MultiLabelAUCMLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p =  imratio 
        self.version = version 
        assert version in ['v1', 'v2'], "Input value is not valid! Possible values are ['v1', 'v2']."
        self.num_labels = num_labels
        if self.p:
           assert len(imratio)==num_labels, 'Length of imratio needs to be same as num_classes!'
        else:
            self.p = [0.0]*num_labels
        self.a = torch.zeros(num_labels, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(num_labels, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_labels, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    @property
    def get_a(self):
        return self.a.mean()
    @property
    def get_b(self):
        return self.b.mean()
    @property
    def get_alpha(self):
        return self.alpha.mean()
    
    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true, task_id=[], auto=True, **kwargs):
        total_loss = 0
        if len(task_id) == 0:
            task_id = range(self.num_labels)
        else:
           task_id = torch.unique(task_id)
        for idx in task_id:
            y_pred_i, y_true_i = y_pred[:, idx].reshape(-1, 1),  y_true[:, idx].reshape(-1, 1)
            if self.version == 'v1':
                if auto or self.p == None:
                   self.p[idx] = (y_true_i==1).sum()/y_true_i.shape[0]   
                loss = (1-self.p[idx])*torch.mean((y_pred_i - self.a[idx])**2*(1==y_true_i).float()) + \
                            self.p[idx]*torch.mean((y_pred_i - self.b[idx])**2*(0==y_true_i).float())   + \
                            2*self.alpha[idx]*(self.p[idx]*(1-self.p[idx]) + \
                            torch.mean((self.p[idx]*y_pred_i*(0==y_true_i).float() - (1-self.p[idx])*y_pred_i*(1==y_true_i).float())) )- \
                            self.p[idx]*(1-self.p[idx])*self.alpha[idx]**2
            else:
                loss = self.mean((y_pred_i - self.a[idx])**2*(1==y_true_i).float()) + \
                            self.mean((y_pred_i - self.b[idx])**2*(0==y_true_i).float())   + \
                            2*self.alpha[idx]*(self.margin  + self.mean(y_pred_i*(0==y_true_i).float()) - self.mean(y_pred_i*(1==y_true_i).float()) )- \
                            self.alpha[idx]**2
            total_loss += loss
        return total_loss/len(task_id)
    

class meanAveragePrecisionLoss(torch.nn.Module):
    r"""
        Mean Average Precision loss based on squared-hinge surrogate loss to optimize mAP and mAP@k. This is an extension of :obj:`~libauc.losses.APLoss`.
        
        Args:
            data_len (int):  total number of samples in the training dataset.
            num_labels (int): number of unique labels(tasks) in the dataset.
            margin (float, optional): margin for the squared-hinge surrogate loss (default: ``1.0``).
            gamma (float, optional): parameter for the moving average estimator (default: ``0.9``).
            top_k (int, optional): If given, only top k items will be considered for optimizing mAP@k.
            surr_loss (str, optional): type of surrogate loss to use. Choices are 'squared_hinge', 'squared', 
                                    'logistic', 'barrier_hinge' (default: ``'squared_hinge'``).
    
        This class is also aliased as :obj:`~libauc.losses.auc.mAPLoss`.

        Example:
            >>> loss_fn = meanAveragePrecisionLoss(data_len=data_length, margin=1.0, num_labels=10, gamma=0.9)
            >>> y_pred = torch.randn((32,10), requires_grad=True)
            >>> y_true = torch.empty((32,10), dtype=torch.long).random_(2)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> task_ids = torch.randint(10, (32,), requires_grad=False)
            >>> loss = loss_fn(y_pred, y_true, index, task_ids)
            >>> loss.backward()

        Reference:
            .. [5] Zhuoning Yuan, Dixian Zhu, Zi-Hao Qiu, Gang Li, Xuanhui Wang, Tianbao Yang.
                   "LibAUC: A Deep Learning Library for X-Risk Optimization."
                   29th SIGKDD Conference on Knowledge Discovery and Data Mining.
                   https://arxiv.org/abs/2306.03065
    """
    def __init__(self, 
                 data_len, 
                 num_labels, 
                 margin=1.0, 
                 gamma=0.9, 
                 top_k=-1, 
                 surr_loss='squared_hinge',  
                 device=None):
        super(meanAveragePrecisionLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.num_labels = num_labels
        self.u_all = torch.zeros((num_labels, data_len, 1)).to(self.device).detach()
        self.u_pos = torch.zeros((num_labels, data_len, 1)).to(self.device).detach()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = get_surrogate_loss(surr_loss)
        self.top_k = top_k

    def forward(self, y_pred, y_true, index, task_id=[], **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, self.num_labels))
        y_true = check_tensor_shape(y_true, (-1, self.num_labels))
        index  = check_tensor_shape(index, (-1,))
        if len(task_id) == 0:
           task_id = list(range(self.num_labels))
        else:
           task_id = torch.unique(task_id)
        total_loss = 0
        for idx in task_id:
            y_pred_i, y_true_i = y_pred[:, idx].reshape(-1, 1),  y_true[:, idx].reshape(-1, 1)
            pos_mask = (1==y_true_i).squeeze()
            assert sum(pos_mask) > 0, 'input data contains no positive sample. To fix it, please use libauc.sampler.TriSampler to resampling data!'
            if len(index) == len(y_pred): 
                index_i = index[pos_mask]   # for positive samples only   
            f_ps  = y_pred_i[pos_mask]      # shape: (len(f_ps), 1)
            f_all = y_pred_i.squeeze()      # shape: (len(f_all), )
            sur_loss = self.surrogate_loss(self.margin, (f_ps - f_all)) # shape: (len(f_ps), len(f_all))
            pos_sur_loss = sur_loss * pos_mask
            self.u_all[idx][index_i] = (1 - self.gamma) * self.u_all[idx][index_i]  + self.gamma * (sur_loss.mean(1, keepdim=True)).detach()
            self.u_pos[idx][index_i] = (1 - self.gamma) * self.u_pos[idx][index_i]  + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
            p_i = (self.u_pos[idx][index_i] - (self.u_all[idx][index_i]) * pos_mask) / (self.u_all[idx][index_i] ** 2) # size of p_i: len(f_ps)* len(y_pred)
            if self.top_k > -1:
                selector = torch.sigmoid(self.top_k - sur_loss.sum(dim=0, keepdim=True).clone())
                p_i *= selector
            p_i.detach_()
            loss = torch.mean(p_i * sur_loss)
            total_loss += loss
        return total_loss/len(task_id)


class MultiLabelpAUCLoss(torch.nn.Module):
    r"""
        Partial AUC loss with squared-hinge surrogate loss to optimize multi-label Paritial AUROC. This is an extension of :obj:`~libauc.losses.pAUCLoss`.

        This class is also aliased as :obj:`~libauc.losses.auc.mPAUCLoss`.

        args:
            mode (str): the specific loss function to be used in the backend (default: '1w').
            num_labels (int): number of unique labels(tasks) in the dataset.
            **kwargs: the required arguments for the selected loss function. 

        Example:
            >>> loss_fn = MultiLabelpAUCLoss(data_len=data_length, margin=1.0, num_labels=10)
            >>> y_pred = torch.randn((32,10), requires_grad=True)
            >>> y_true = torch.empty((32,10), dtype=torch.long).random_(2)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> task_ids = torch.randint(10, (32,), requires_grad=False)
            >>> loss = loss_fn(y_pred, y_true, index, task_ids)
            >>> loss.backward()
            
        Reference:
            .. [5] Zhuoning Yuan, Dixian Zhu, Zi-Hao Qiu, Gang Li, Xuanhui Wang, Tianbao Yang.
                   "LibAUC: A Deep Learning Library for X-Risk Optimization."
                   29th SIGKDD Conference on Knowledge Discovery and Data Mining.
                   https://arxiv.org/abs/2306.03065
    """
    def __init__(self, mode='1w', num_labels=10, device=None, **kwargs):
        super(MultiLabelpAUCLoss, self).__init__()
        assert mode in ['SOPA', 'SOPAs', 'SOTAs', '1w', '2w'], 'Keyword is not found!'  #'SOPA', 'SOPAs', 'SOTA'
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.mode = mode 
        self.num_labels = num_labels
        self.loss_fn = self.get_loss(mode, **kwargs)
        self.u_pos_all = torch.zeros((num_labels, self.loss_fn.data_len, 1)).to(self.device).detach()
           
    def get_loss(self, mode='1w', **kwargs):
        if mode == 'SOPA':
           loss = pAUC_CVaR_Loss(**kwargs)
        elif mode == 'SOPAs' or mode=='1w':
           loss = pAUC_DRO_Loss(**kwargs)
        elif mode == 'SOTAs' or mode=='2w':
           loss = tpAUC_KL_Loss(**kwargs)
        else:
            raise ValueError('Out of options!')
        return loss
   
    def forward(self, y_pred, y_true, index, task_id=[], **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, self.num_labels))
        y_true = check_tensor_shape(y_true, (-1, self.num_labels))
        index  = check_tensor_shape(index, (-1,))
        if len(task_id) == 0 : # TODO: add None detector
           task_id = list(range(self.num_labels))
        else:
           task_id = torch.unique(task_id)
        total_loss = 0
        for idx in task_id:
            y_pred_i, y_true_i = y_pred[:, idx].reshape(-1, 1),  y_true[:, idx].reshape(-1, 1)
            self.loss_fn.u_pos = self.u_pos_all[idx] 
            loss = self.loss_fn(y_pred_i, y_true_i, index)
            self.u_pos_all[idx] = self.loss_fn.u_pos 
            total_loss += loss
        return total_loss/len(task_id)

# alias 
APLoss = AveragePrecisionLoss
mAPLoss  = meanAveragePrecisionLoss
mAUCMLoss = MultiLabelAUCMLoss
mPAUCLoss = MultiLabelpAUCLoss
