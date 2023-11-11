import torch 
import torch.nn.functional as F
from .surrogate import get_surrogate_loss
from ..utils.utils import check_tensor_shape

class Top_Push_Loss(torch.nn.Module):
    """
      Partial AUC loss based on Top Push Loss to optimize One-way Partial AUROC (OPAUC).
      
      Args:
        pos_len (int): number of positive examples in the training data
        num_neg (int): number of negative samples for each mini-batch
        margin: margin used in surrogate loss (default: ``squared_hinge``)
        alpha: upper bound of False Positive Rate (FPR) used for optimizing pAUC (default: ``0``).
        beta (float): upper bound of False Positive Rate (FPR) used for optimizing pAUC (default: ``0.2``).
      
    Reference:
        [1] Zhu, Dixian and Li, Gang and Wang, Bokun and Wu, Xiaodong and Yang, Tianbao.
        "When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee."
        In International Conference on Machine Learning, pp. 27548-27573. PMLR, 2022.
        https://proceedings.mlr.press/v162/zhu22g.html
            
    """                                     
    def __init__(self, 
                 pos_len, 
                 num_neg, 
                 margin=1.0, 
                 beta=0.2, 
                 surrogate_loss='squared_hinge'):
        
        super(Top_Push_Loss, self).__init__()                                 
        self.beta = 1/num_neg  # choose hardest negative samples in mini-batch                                
        self.eta = 1.0
        self.num_neg = num_neg
        self.pos_len = pos_len
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).cuda()             
        self.margin = margin                        
        self.surrogate_loss = get_surrogate_loss(surrogate_loss)
                                       
    def forward(self, y_pred, y_true, index, auto=True):
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
        p = loss > self.u_pos[index]
        self.u_pos[index] = self.u_pos[index]-self.eta/self.pos_len*(1 - p.sum(dim=1, keepdim=True)/(self.beta*self.num_neg))
        p.detach_()
        loss = torch.mean(p * loss) / self.beta
        return loss
