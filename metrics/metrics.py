from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
from ..utils.utils import check_array_type, check_tensor_shape, check_array_shape, select_mean


def auc_roc_score(y_true, y_pred, reduction='mean', **kwargs):
    r"""Evaluation function of AUROC"""
    y_true = check_array_type(y_true)
    y_pred = check_array_type(y_pred)
    num_labels = y_true.shape[-1] if len(y_true) == 2 else 1
    y_true = check_array_shape(y_true, (-1, num_labels)) 
    y_pred = check_array_shape(y_pred, (-1, num_labels))
    assert reduction in ['mean', None, 'None'], 'Input is not valid!'
    if y_pred.shape[-1] != 1 and len(y_pred.shape) > 1:
        class_auc_list = []
        for i in range(y_pred.shape[-1]):
            try:
                local_auc = roc_auc_score(y_true[:, i], y_pred[:, i],  **kwargs)
                class_auc_list.append(local_auc)
            except: 
                # edge case: no positive samples in the data set
                class_auc_list.append(-1.0) # if only one class
        if reduction == 'mean':
            return select_mean(class_auc_list, threshold=0) # return non-negative mean
        return class_auc_list
    return roc_auc_score(y_true, y_pred, **kwargs)


def auc_prc_score(y_true, y_pred, reduction='mean', **kwargs):
    r"""Evaluation function of AUPRC"""
    y_true = check_array_type(y_true)
    y_pred = check_array_type(y_pred)
    num_labels = y_true.shape[-1] if len(y_true) == 2 else 1
    y_true = check_array_shape(y_true, (-1, num_labels)) 
    y_pred = check_array_shape(y_pred, (-1, num_labels))
    if y_pred.shape[-1] != 1 and len(y_pred.shape)>1:
        class_auc_list = []
        for i in range(y_pred.shape[-1]):
            try:
                local_auc = average_precision_score(y_true[:, i], y_pred[:, i])
                class_auc_list.append(local_auc)
            except: 
                # edge case: no positive samples in the data set
                class_auc_list.append(-1.0)
        if reduction == 'mean':
            return select_mean(class_auc_list)
        return class_auc_list
    return average_precision_score(y_true, y_pred, **kwargs)


def pauc_roc_score(y_true, y_pred, max_fpr=1.0, min_tpr=0.0, reduction='mean', **kwargs):
    r"""Evaluation function of pAUROC"""
    y_true = check_array_type(y_true)
    y_pred = check_array_type(y_pred)
    #num_labels = y_true.shape[-1] if len(y_true) == 2 else 1
    y_true = check_array_shape(y_true, (-1,)) 
    y_pred = check_array_shape(y_pred, (-1,))

    # TODO: multi-label support 
    if min_tpr == 0:
        # One-way Partial AUC (OPAUC)
        return roc_auc_score(y_true, y_pred, max_fpr=max_fpr, **kwargs)

    # Two-way Partial AUC (TPAUC)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true != 1)[0]
    num_pos = round(len(pos_idx)*(1-min_tpr))
    num_neg = round(len(neg_idx)*max_fpr)
    num_pos = 1 if num_pos < 1 else num_pos
    num_neg = 1 if num_neg < 1 else num_neg
    if len(pos_idx)==1: 
        selected_pos = [0]
    else:
        selected_pos = np.argpartition(y_pred[pos_idx], num_pos)[:num_pos]
    if len(neg_idx)==1: 
        selected_neg = [0]
    else:
        selected_neg = np.argpartition(-y_pred[neg_idx], num_neg)[:num_neg]
    selected_target = np.concatenate((y_true[pos_idx][selected_pos], y_true[neg_idx][selected_neg]))
    selected_pred = np.concatenate((y_pred[pos_idx][selected_pos], y_pred[neg_idx][selected_neg]))
    return roc_auc_score(selected_target, selected_pred, **kwargs)


# Reference: https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code
def precision_and_recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs):
    # referece: https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
    def calc_metrics(y_true, y_pred):
        y_true = y_true == pos_label 
        desc_sort_order = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[desc_sort_order]
        true_positives = y_true_sorted[:k].sum()
        total_positives = sum(y_true)

        precision_k = true_positives / min(k, total_positives)
        recall_k = true_positives / total_positives
        return precision_k, recall_k

    y_true = check_array_shape(y_true, (-1, 1))
    y_pred = check_array_shape(y_pred, (-1, 1))

    if y_true.shape[-1] != 1 and len(y_true.shape) > 1:
        metrics_list = [calc_metrics(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[-1])]
        precision_k_list, recall_k_list = zip(*metrics_list)
        return precision_k_list, recall_k_list
    else:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision_k, recall_k = calc_metrics(y_true, y_pred)
        return precision_k, recall_k

def precision_at_k(y_true, y_pred, k, pos_label=1, **kwargs):
    r"""Evaluation function of Precision@K"""
    precision_k, _ = precision_and_recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs)
    return precision_k

def recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs):
    r"""Evaluation function of Recall@K"""
    _, recall_k = precision_and_recall_at_k(y_true, y_pred, k, pos_label=1, **kwargs)
    return recall_k

def ap_at_k(y_true, y_pred, k=10):
    r"""Evaluation function of AveragePrecision@K"""
    # adapted from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    y_true = check_array_shape(y_true, (-1,))
    y_pred = check_array_shape(y_pred, (-1,))
    if len(y_pred)>k:
        y_pred = y_pred[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score / min(len(y_true), k)

def map_at_k(y_true, y_pred, k=10):
    r"""Evaluation function of meanAveragePrecision@K"""
    # adapted from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    assert len(y_true.shape) == 2 and len(y_true.shape) == 2 
    assert k > 0, 'Value of k is not valid!'
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    return np.mean([ap_at_k(a,p,k) for a,p in zip(y_true, y_pred)])


def ndcg_at_k(y_true, y_pred, k=5):
    r"""
        Evaluation function of NDCG@K
    """
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert len(y_pred.shape) == 2 and len(y_pred.shape) == 2

    num_of_users, num_pos_items = y_true.shape
    sorted_ratings = -np.sort(-y_true)            # descending order !!
    discounters = np.tile([np.log2(i+1) for i in range(1, 1+num_pos_items)], (num_of_users, 1))
    normalizer_mat = (np.exp2(sorted_ratings) - 1) / discounters
    
    sort_idx = (-y_pred).argsort(axis=1)    # index of sorted predictions (max->min)
    gt_rank = np.array([np.argwhere(sort_idx == i)[:, 1]+1 for i in range(num_pos_items)]).T  # rank of the ground-truth (start from 1)
    hit = (gt_rank <= k)
    
    # calculate the normalizer first
    normalizer = np.sum(normalizer_mat[:, :k], axis=1)
    # calculate DCG
    DCG = np.sum(((np.exp2(y_true) - 1) / np.log2(gt_rank+1)) * hit.astype(float), axis=1)
    return np.mean(DCG / normalizer)

# TODO: automatic detect classificaiton task or ranking task?
def evaluator(y_true, y_pred, metrics=['auroc', 'auprc', 'pauroc'], return_str=False, format='%.4f(%s)', **kwargs):
    results = {}
    if 'auroc' in metrics:
      results['auroc'] = auc_roc_score(y_true, y_pred) 
    if 'auprc' in metrics:
      results['auprc'] = auc_prc_score(y_true, y_pred)      
    if 'pauroc' in metrics:
      results['pauroc'] = pauc_roc_score(y_true, y_pred, **kwargs)   # e.g., max_fpr=0.3
    if return_str:
      output = []
      for key, value in results.items():
          output.append(format%(value, key))
      return ','.join(output)    
    return results


if __name__ == '__main__':
    # import numpy as np
    preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0]

    print (roc_auc_score(labels, preds))
    print (average_precision_score(labels, preds))


