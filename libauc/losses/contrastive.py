import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


class GCLoss_v1(nn.Module):
    r"""
        Stochastic Optimization of Global Contrastive Loss (GCL) and Robust Global Contrastive Loss (RGCL) for learning representations for unimodal tasks (e.g., image-image). The objective for optimizing GCL (i.e., objective for SogCLR) is defined as

        .. math::

            F(\mathbf{w}) = \frac{1}{n} \sum_{\mathbf{x}_i \in D} { \tau \log \mathbb{E}_{\mathbf{z}\in S_i^-} \exp \Big( \frac{h_i(\mathbf{z})}{\tau} \Big) },

        and the objective for optimizing RGCL (i.e., objective for iSogCLR) is defined as

        .. math::

            F(\mathbf{w},\mathbf{\tau}) = \frac{1}{n} \sum_{\mathbf{x}_i \in D} { \mathbf{\tau}_i \log \mathbb{E}_{\mathbf{z}\in S_i^-} \exp \Big( \frac{h_i(\mathbf{z})}{\mathbf{\tau}_i} \Big) + \mathbf{\tau}_i \rho },

        where :math:`h_i(\mathbf{z})=E(\mathcal{A}(\mathbf{x}_i))^{\mathrm{T}}E(\mathbf{z})-E(\mathcal{A}(\mathbf{x}_i))^{\mathrm{T}}E(\mathcal{A}^{\prime}(\mathbf{x}_i))`, :math:`\mathcal{A}` and :math:`\mathcal{A}^{\prime}` are two data
        augmentation operations, :math:`S_i^-` denotes all negative samples for anchor data :math:`\mathbf{x}_i`, and :math:`E(\cdot)` represents the image encoder. In iSogCLR, :math:`\mathbf{\tau}_i` is the individualized
        temperature for :math:`\mathbf{x}_i`.

        Args:
            N (int): number of samples in the training dataset (default: ``100000``)
            tau (float): temperature parameter for global contrastive loss. If you enable isogclr, then input temperature will be the initial value for learnable temperature parameters (default: ``0.1``)
            device (torch.device): the device for the inputs (default: ``None``)
            distributed (bool): whether to use distributed training (default: ``False``)
            enable_isogclr (bool, optional): whether to enable iSogCLR. If True, then the algorithm will optimize individualized temperature parameters for all samples (default: ``False``)
            eta (float, optional): the step size for updating temperature parameters in iSogCLR (default: ``0.01``)
            rho (float, optional): the hyperparameter :math:`\rho` in Eq. (6) in iSogCLR [2] (default: ``0.3``)
            tau_min (float, optional): lower bound of learnable temperature in iSogCLR (default: ``0.05``)
            tau_max (float, optional): upper bound of learnable temperature in iSogCLR (default: ``0.7``)
            beta (float, optional): the momentum parameter for updating temperature parameters in iSogCLR (default: ``0.9``)

        Example:
            >>> loss_fn = GCLoss_v1(N=1000, tau=0.1)
            >>> img_feat1, img_feat2 = torch.randn((32, 256), requires_grad=True), torch.randn((32, 256), requires_grad=True)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(img_feat1, img_feat2, index)
            >>> loss.backward()

        Reference:
            .. [1] Yuan, Z., Wu, Y., Qiu, Z., Du, X., Zhang, L., Zhou, D., and Yang, T.
                   Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance
                   https://arxiv.org/abs/2202.12387

            .. [2] Qiu, Z., Hu, Q., Yuan, Z., Zhou, D., Zhang, L., and Yang, T.
                   Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization
                   https://arxiv.org/abs/2305.11965
    """
    def __init__(self, 
                 N=100000, 
                 tau=0.1,
                 gamma=0.9,  
                 eps=1e-8,
                 device=None, 
                 distributed=False,
                 enable_isogclr=False,
                 tau_min=0.05, tau_max=0.7, rho=0.3, eta=0.01, beta=0.9):
        super(GCLoss_v1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.N = N
        self.u = torch.zeros(N).reshape(-1, 1) #.to(self.device) 
        self.tau = tau
        self.gamma = gamma 
        self.distributed = distributed
        self.LARGE_NUM = 1e9
        self.eps = eps

        # settings for isogclr
        self.enable_isogclr = enable_isogclr
        if self.enable_isogclr:
            self.tau_min, self.tau_max = tau_min, tau_max     # lower and upper bound for learnable tau
            self.rho = rho                                    # tunable parameter for isogclr, recommended values for unimodal tasks: [0.1~0.5]
            self.eta = eta                                    # learning rate for learnable tau
            self.beta = beta                                  # momentum parameter for the gradients of learnable tau
            self.learnable_tau = torch.ones(N).reshape(-1, 1) * self.tau
            self.grad_tau = torch.zeros(N).reshape(-1, 1)

    def forward(self, 
               hidden1, 
               hidden2, 
               index):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
        
        # Gather hidden1/hidden2 across replicas and create local labels.
        if self.distributed:
           hidden1_large, hidden2_large = gather_features(hidden1, hidden2)
           enlarged_batch_size = hidden1_large.shape[0]

           labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size  * torch.distributed.get_rank()).to(self.device) 
           labels = F.one_hot(labels_idx, enlarged_batch_size*2).to(self.device) 
           masks  = F.one_hot(labels_idx, enlarged_batch_size).to(self.device) 
           batch_size = enlarged_batch_size
        else:
           hidden1_large = hidden1
           hidden2_large = hidden2
           labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
           masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa], 1) 
        logits_ba_bb = torch.cat([logits_ba, logits_bb], 1)

        if self.enable_isogclr:
            tau = self.learnable_tau[index].cuda()
            neg_logits1 = torch.exp(logits_ab_aa/tau[:, None])*neg_mask   #(B, 2B)
            neg_logits2 = torch.exp(logits_ba_bb/tau[:, None])*neg_mask

        else:
            neg_logits1 = torch.exp(logits_ab_aa/self.tau)*neg_mask   #(B, 2B)
            neg_logits2 = torch.exp(logits_ba_bb/self.tau)*neg_mask

        # u init    
        if self.u[index].sum() == 0:
            u1 = torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
            u2 = torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))
        else:
            u1 = (1 - self.gamma ) * self.u[index].cuda() + self.gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
            u2 = (1 - self.gamma ) * self.u[index].cuda() + self.gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))

        # this sync on all devices (since "hidden" are gathering from all devices)  #### maybe we can concat_all_gather index before?
        if self.distributed:
           u1_large = concat_all_gather(u1)
           u2_large = concat_all_gather(u2)
           index_large = concat_all_gather(index)
           self.u[index_large] =  (u1_large.detach().cpu() + u2_large.detach().cpu())/2
        else:
           self.u[index] = (u1.detach().cpu() + u2.detach().cpu())/2

        p_neg_weights1 = (neg_logits1/u1.clamp(min=self.eps)).detach()
        p_neg_weights2 = (neg_logits2/u2.clamp(min=self.eps)).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        if self.enable_isogclr:
            # update learnable temperature parameters
            grad_tau_a = torch.log(u1) + self.rho - torch.sum(p_neg_weights1 * (logits_ab_aa/tau[:, None]).detach(), dim=1, keepdim=True)/(2*(batch_size-1))
            grad_tau_b = torch.log(u2) + self.rho - torch.sum(p_neg_weights2 * (logits_ba_bb/tau[:, None]).detach(), dim=1, keepdim=True)/(2*(batch_size-1))

            grad_tau = (grad_tau_a + grad_tau_b) / 2.0

            self.grad_tau[index] = (1.0 - self.beta) * self.grad_tau[index] + self.beta * grad_tau.cpu()
            self.learnable_tau[index] = (self.learnable_tau[index] - self.eta * self.grad_tau[index]).clamp_(min=self.tau_min, max=self.tau_max)

        return loss

    

class GCLoss_v2(nn.Module):
    r"""
        Stochastic Optimization of Global Contrastive Loss (GCL) and Robust Global Contrastive Loss (RGCL) for learning
        representations for bimodal task (e.g., image-text). The objective for optimizing GCL (i.e., objective for SogCLR) is defined as

        .. math::

            F(\mathbf{w}) = \frac{1}{n} \sum_{(\mathbf{x}_i, \mathbf{t}_i) \in D} { \tau \log \mathbb{E}_{\mathbf{t}\in T_i^-} \exp \Big( \frac{h_{\mathbf{x}_i}(\mathbf{t})}{\tau} \Big) + \tau \log \mathbb{E}_{\mathbf{x}\in I_i^-} \exp \Big( \frac{h_{\mathbf{t}_i}(\mathbf{x})}{\tau} \Big) },

        and the objective for optimizing RGCL (i.e., objective for iSogCLR) is defined as

        .. math::

            F(\mathbf{w}, \mathbf{\tau}, \mathbf{\tau}^{\prime}) = \frac{1}{n} \sum_{(\mathbf{x}_i, \mathbf{t}_i) \in D} { (\mathbf{\tau}_i + \mathbf{\tau}^{\prime}_i)\rho + \mathbf{\tau}_i \log \mathbb{E}_{\mathbf{t}\in T_i^-} \exp \Big( \frac{h_{\mathbf{x}_i}(\mathbf{t})}{\mathbf{\tau}_i} \Big) + \mathbf{\tau}^{\prime}_i \log \mathbb{E}_{\mathbf{x}\in I_i^-} \exp \Big( \frac{h_{\mathbf{t}_i}(\mathbf{x})}{\mathbf{\tau}^{\prime}_i} \Big) },

        where :math:`(\mathbf{x}_i, \mathbf{t}_i) \in D` is an image-text pair, :math:`h_{\mathbf{x}_i}(\mathbf{t})=E_I(\mathbf{x}_i)^{\mathrm{T}}E_T(\mathbf{t}) - E_I(\mathbf{x}_i)^{\mathrm{T}}E_T(\mathbf{t}_i)`, :math:`h_{\mathbf{t}_i}(\mathbf{x})=E_I(\mathbf{x})^{\mathrm{T}}E_T(\mathbf{t}_i) - E_I(\mathbf{x}_i)^{\mathrm{T}}E_T(\mathbf{t}_i)`, 
        :math:`E_I(\cdot)` and :math:`E_T(\cdot)` are image and text encoder, respectively. In iSogCLR, :math:`\mathbf{\tau}_i`, :math:`\mathbf{\tau}^{\prime}_i` are individualized temperature for :math:`\mathbf{x}_i` and :math:`\mathbf{t}_i`, respectively.


        Args:
            N (int): number of samples in the training dataset (default: ``100000``)
            tau (float): temperature parameter for global contrastive loss. If you enable isogclr, then input temperature will be the initial value for learnable temperature parameters (default: ``0.1``)
            gamma (float): the moving average factor for dynamic loss in range the range of (0.0, 1.0) (default: ``0.9``)
            cache_labels (bool): whether to cache labels for mini-batch data (default: ``True``)
            rank (int):  unique ID given to a process for distributed training (default: ``0``)
            world_size (int): total number of processes for distributed training (default: ``1``)
            distributed (bool): whether to use distributed training (default: ``False``)
            enable_isogclr (bool, optional): whether to enable iSogCLR. If True, then the algorithm will optimize individualized temperature parameters for all samples (default: ``False``)
            eta (float, optional): the step size for updating temperature parameters in iSogCLR (default: ``0.01``)
            rho (float, optional): the hyperparameter :math:`\rho` in Eq. (6) in iSogCLR [2] (default: ``6.0``)
            tau_min (float, optional): lower bound of learnable temperature in iSogCLR (default: ``0.005``)
            tau_max (float, optional): upper bound of learnable temperature in iSogCLR (default: ``0.05``)
            beta (float, optional): the momentum parameter for updating temperature parameters in iSogCLR (default: ``0.9``)


        Example:
            >>> loss_fn = GCLoss_v2(N=1000, tau=0.1)
            >>> img_feat, txt_feat = torch.randn((32, 256), requires_grad=True), torch.randn((32, 256), requires_grad=True)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(img_feat, txt_feat, index)
            >>> loss.backward()

        Reference:
            .. [1] Yuan, Z., Wu, Y., Qiu, Z., Du, X., Zhang, L., Zhou, D., and Yang, T.
                   Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance
                   https://arxiv.org/abs/2202.12387.

            .. [2] Qiu, Z., Hu, Q., Yuan, Z., Zhou, D., Zhang, L., and Yang, T.
                   Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization
                   https://arxiv.org/abs/2305.11965.
    """
    def __init__(
            self,
            N=1000000,
            tau=0.01, 
            gamma=0.9,
            cache_labels=False,
            rank=0,
            world_size=1,
            distributed=False,
            enable_isogclr=False,
            tau_min=0.005, tau_max=0.05, rho=6.0, eta=0.01, beta=0.9):
        super(GCLoss_v2, self).__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.distributed = distributed

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.world_size = world_size
        
        self.b1 = torch.zeros(N).reshape(-1, 1).detach() # avoid overflow when tau is small
        self.b2 = torch.zeros(N).reshape(-1, 1).detach()
        
        # sogclr        
        self.u1 = torch.zeros(N).reshape(-1, 1).detach()
        self.u2 = torch.zeros(N).reshape(-1, 1).detach()
        self.gamma = gamma 
        self.tau = tau

        self.eps = 1e-20

        # setting for isogclr
        self.enable_isogclr = enable_isogclr
        if self.enable_isogclr:
            self.tau_min, self.tau_max = tau_min, tau_max
            self.rho = rho
            self.eta = eta
            self.beta = beta
            self.learnable_tau_img = torch.ones(N).reshape(-1, 1) * self.tau
            self.learnable_tau_txt = torch.ones(N).reshape(-1, 1) * self.tau
            self.grad_tau_img = torch.zeros(N).reshape(-1, 1)
            self.grad_tau_txt = torch.zeros(N).reshape(-1, 1)

    def forward(self, image_features, text_features, index):
        device = image_features.device
        
        if self.distributed:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.rank, self.world_size)

            logits_per_image =  image_features @ all_text_features.T   
            logits_per_text  =  text_features  @ all_image_features.T  

        else:
            logits_per_image = image_features @ text_features.T   
            logits_per_text  = logits_per_image.T                 
            
        logits_image = logits_per_image - torch.diagonal(logits_per_image)[:,None]
        logits_text = logits_per_text - torch.diagonal(logits_per_text)[:,None]
            
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long) 
            if self.distributed:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # insert sogclr code here
        large_batch_size = logits_per_image.shape[-1]
        labels_onehot = F.one_hot(labels, large_batch_size)    
        neg_mask = 1 - labels_onehot
        
        if self.enable_isogclr:
            tau_img = self.learnable_tau_img[index].to(device)
            tau_txt = self.learnable_tau_txt[index].to(device)
        else:
            tau_img = self.tau
            tau_txt = self.tau
            
        logits_image_d_tau = (logits_image / tau_img).detach()
        logits_text_d_tau = (logits_text / tau_txt).detach()
        
        old_b1 = self.b1[index].to(device)
        new_b1 = torch.max(logits_image_d_tau, old_b1.tile(1, large_batch_size))
        self.b1[index] = torch.max(new_b1, dim=1, keepdim=True)[0].cpu()
        
        old_b2 = self.b2[index].to(device)
        new_b2 = torch.max(logits_text_d_tau, old_b2.tile(1, large_batch_size))
        self.b2[index] = torch.max(new_b2, dim=1, keepdim=True)[0].cpu()
        
        neg_logits_image = torch.exp(logits_image_d_tau - self.b1[index].to(device))*neg_mask   #(B, 4B)
        neg_logits_text  = torch.exp(logits_text_d_tau - self.b2[index].to(device))*neg_mask    #(B, 4B)
        
        # u init    
        if self.u1[index].sum() == 0:
            u1 = torch.sum(neg_logits_image, dim=1, keepdim=True)/(large_batch_size-1)
        else:
            u1 = (1 - self.gamma) * self.u1[index].to(device) * torch.exp(old_b1 - self.b1[index].to(device)) \
                     + self.gamma * torch.sum(neg_logits_image, dim=1, keepdim=True)/(large_batch_size-1)
        
        if self.u2[index].sum() == 0:
            u2 = torch.sum(neg_logits_text, dim=1, keepdim=True)/(large_batch_size-1)
        else:
            u2 = (1 - self.gamma) * self.u2[index].to(device) * torch.exp(old_b2 - self.b2[index].to(device)) \
                     + self.gamma * torch.sum(neg_logits_text, dim=1, keepdim=True)/(large_batch_size-1)

        u1 = u1.clamp(min=self.eps)
        u2 = u2.clamp(min=self.eps)
        
        p1 = (neg_logits_image/u1).detach()
        p2 = (neg_logits_text/u2).detach()
        
        if self.world_size > 1:
            gathered_u1 = concat_all_gather(u1)  # [global_batch size, 1]
            gathered_u2 = concat_all_gather(u2)
            index_large = concat_all_gather(index)
            self.u1[index_large] =  gathered_u1.detach().cpu()
            self.u2[index_large] =  gathered_u2.detach().cpu()
        else:
            self.u1[index] = u1.detach().cpu()
            self.u2[index] = u2.detach().cpu()
             
        img_loss = torch.sum(p1 * logits_image, dim=1, keepdim=True).mean()
        txt_loss = torch.sum(p2 * logits_text, dim=1, keepdim=True).mean()
         
        total_loss = (img_loss + txt_loss)/2

        if self.enable_isogclr:
            grad_tau_img = torch.log(u1.detach()) + self.rho + self.b1[index].to(device) \
                            - torch.sum(p1 * logits_image_d_tau, dim=1, keepdim=True)/(large_batch_size-1)
            grad_tau_txt = torch.log(u2.detach()) + self.rho + self.b2[index].to(device) \
                            - torch.sum(p2 * logits_text_d_tau, dim=1, keepdim=True)/(large_batch_size-1)

            self.grad_tau_img[index] = (1.0 - self.beta) * self.grad_tau_img[index] + self.beta * grad_tau_img.cpu()
            self.grad_tau_txt[index] = (1.0 - self.beta) * self.grad_tau_txt[index] + self.beta * grad_tau_txt.cpu()
            self.learnable_tau_img[index] = (self.learnable_tau_img[index] - self.eta * self.grad_tau_img[index]).clamp_(min=self.tau_min, max=self.tau_max)
            self.learnable_tau_txt[index] = (self.learnable_tau_txt[index] - self.eta * self.grad_tau_txt[index]).clamp_(min=self.tau_min, max=self.tau_max)

            return total_loss, (tau_img.mean(), tau_txt.mean())

        return total_loss, None


    
class GCLoss(torch.nn.Module):
    r"""
        A high-level wrapper for :class:`~libauc.losses.contrastive.GCLoss_v1`  and :class:`~libauc.losses.contrastive.GCLoss_v2`.

        Args:
            mode (str, optional): type of GCLoss to use. Options are 'unimodal' for GCLoss_v1 and 'bimodal' for GCLoss_v2 (default: ``'unimodal'``).
            **kwargs: arbitrary keyword arguments. These will be passed directly to the chosen GCLoss version's constructor.

        Example:
            >>> loss_fn = GCLoss(mode='bimodal', N=1000, tau=0.1)
            >>> feat_img, feat_txt = torch.randn((32, 256), requires_grad=True), torch.randn((32, 256), requires_grad=True)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> dynamic_loss = loss_fn(feat_img=feat_img, feat_txt=feat_txt, index=index)

        Note:
            The forward method of this class simply calls the forward method of the chosen GCLoss (:obj:`~libauc.losses.contrastive.GCLoss_v1` or :obj:`~libauc.losses..ontrastive.GCLoss_v2`).
    """
    def __init__(self, mode='unimodal', **kwargs):
        super(GCLoss, self).__init__()
        assert mode in ['unimodal', 'bimodal'], 'Keyword is not found!'
        self.mode = mode 
        self.loss_fn = self.get_loss(mode, **kwargs)   
                        
    def get_loss(self, mode='unimodal', **kwargs):
        if mode == 'unimodal':
           loss = GCLoss_v1(**kwargs)
        elif mode == 'bimodal':
           loss = GCLoss_v2(**kwargs)
        else:
            raise ValueError('Out of options!')
        return loss
    
    def forward(self, hidden1, hidden2, index,  **kwargs):
        return self.loss_fn(hidden1, hidden2, index, **kwargs)
    

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors. ***Warning ***: torch.distributed.all_gather has no gradient."""
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class all_gather_layer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return 


def gather_features(image_features, text_features, rank=0, world_size=1):
    """We gather tensors from all gpus"""
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
    all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    return all_image_features, all_text_features