import torch 
import torch.nn.functional as F
from ..utils.utils import check_tensor_shape


class CrossEntropyLoss(torch.nn.Module):
    r"""
        Cross-Entropy loss with a sigmoid function. This implementation is based on the built-in function 
        from :obj:`~torch.nn.functional.binary_cross_entropy_with_logits`. 

        Example:
            >>> loss_fn = CrossEntropyLoss()
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> loss = loss_fn(preds, target)
            >>> loss.backward()

        Reference: 
            https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits  # with sigmoid

    def forward(self, y_pred, y_true): # TODO: handle the tensor shapes
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))         
        return self.criterion(y_pred, y_true)
    
class FocalLoss(torch.nn.Module):
    r"""
    Focal loss with a sigmoid function.
    
    Args:
        alpha (float): weighting factor in range (0,1) to balance positive vs negative examples (Default: ``0.25``).
        gamma (float): exponent of the modulating factor (1 - p_t) to balance easy vs hard examples (Default: ``2``).
    
    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> preds = torch.randn(32, 1, requires_grad=True)
        >>> target = torch.empty(32, dtype=torch.long).random_(1)
        >>> loss = loss_fn(preds, target)
        >>> loss.backward() 

    Reference: 
        .. [1] Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr. 
                "Focal loss for dense object detection." 
                Proceedings of the IEEE international conference on computer vision. 2017.
    """
    def __init__(self, alpha=.25, gamma=2, device=None):
        super(FocalLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device  
        self.alpha = torch.tensor([alpha, 1-alpha]).to(self.device)
        self.gamma = torch.tensor([gamma]).to(self.device)

    def forward(self, y_pred, y_true):
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))     
        BCE_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        y_true = y_true.type(torch.long)
        at = self.alpha.gather(0, y_true.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

    
