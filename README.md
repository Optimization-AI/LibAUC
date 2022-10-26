<p align="center">
  <img src="https://github.com/yzhuoning/LibAUC/blob/main/imgs/libauc.png" width="70%" align="center"/>
</p>
<p align="center">
  Logo by <a href="https://zhuoning.cc">Zhuoning Yuan</a>
</p>

**LibAUC**: A Deep Learning Library for X-Risk Optimization
---
<p align="left">
  <img alt="PyPI version" src="https://img.shields.io/pypi/v/libauc?color=blue&style=flat"/>
  <img alt="PyPI version" src="https://static.pepy.tech/personalized-badge/libauc?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"/>
  <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/libauc?color=blue&style=flat" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.8-yellow?color=blue&style=flat" />	
  <img alt="PyPI LICENSE" src="https://img.shields.io/github/license/yzhuoning/libauc?color=blue&logo=libauc&style=flat" />	
</p>


[**Website**](https://libauc.org/)
| [**Updates**](https://libauc.org/news/)
| [**Installation**](https://libauc.org/installation/)
| [**Tutorial**](https://github.com/Optimization-AI/LibAUC/tree/main/examples)
| [**Research**](https://libauc.org/publications/)
| [**Github**](https://github.com/Optimization-AI/LibAUC/)

We continuously update our library by making improvements and adding new features. If you use or like our library, please **star**:star: this repo. Thank you!


:calendar: Updates
---
- **2022/7**: LibAUC **1.2.0** is released! In this version, we've included more losses and optimizers as well as made some performance improvements. Please check [release note](https://github.com/Optimization-AI/LibAUC/releases/tag/v1.2.0) for more details! Thanks!


:mag: What is X-Risk?
---
X-risk refers to a family of compositional measures/losses, in which each data point is compared with a set of data points explicitly or implicitly for defining a risk function. It covers a family of widely used measures/losses including but not limited to the following four interconnected categories:
- **[Areas under the curves]()**, including areas under ROC curves (AUROC), areas under Precision-Recall curves (AUPRC), one-way and two-wary partial areas under ROC curves.
- **[Ranking measures/objectives]()**, including p-norm push for bipartite ranking, listwise losses for learning to rank (e.g., listNet), mean average precision (mAP), normalized discounted cumulative gain (NDCG), etc.
- **[Performance at the top]()**, including top push, top-K variants of mAP and NDCG, Recall at top K positions (Rec@K), Precision at a certain Recall level (Prec@Rec), etc.
- **[Contrastive objectives]()**, including supervised contrastive objectives (e.g., NCA), and global self-supervised contrastive objectives improving upon SimCLR and CLIP.


:star: Key Features
---
- **[Easy Installation](https://github.com/Optimization-AI/LibAUC#key-features)** - Easy to install and insert LibAUC code into existing training pipeline with Deep Learning frameworks like PyTorch.
- **[Broad Applications](https://github.com/Optimization-AI/LibAUC#key-features)** - Users can learn different neural network structures (e.g., MLP, CNN, GNN, transformer, etc) that support their data types.
- **[Efficient Algorithms](https://github.com/Optimization-AI/LibAUC#key-features)** - Stochastic algorithms with provable theoretical convergence that support learning with millions of data points without larger batch size.
- **[Hands-on Tutorials](https://github.com/Optimization-AI/LibAUC#key-features)** - Hands-on tutorials are provided for optimizing a variety of measures and objectives belonging to the family of X-risks.


:gear: Installation
--------------
```
$ pip install libauc==1.2.0
```
The latest version **`1.2.0`** is available now! You can check [release note](https://github.com/Optimization-AI/LibAUC/releases/tag/v1.2.0) for more details. Source code is available for download [here](https://github.com/Optimization-AI/LibAUC/releases). 


:clipboard: Usage
---
#### Example training pipline for optimizing X-risk (e.g., AUROC) 
```python
>>> #import our loss and optimizer
>>> from libauc.losses import AUCMLoss 
>>> from libauc.optimizers import PESG 
...
>>> #define loss & optimizer
>>> Loss = AUCMLoss()
>>> optimizer = PESG()
...
>>> #training
>>> model.train()    
>>> for data, targets in trainloader:
>>>	data, targets  = data.cuda(), targets.cuda()
        logits = model(data)
	preds = torch.sigmoid(logits)
        loss = Loss(preds, targets) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
...	
>>> #update internal parameters
>>> optimizer.update_regularizer()
```

:notebook_with_decorative_cover: Tutorials
-------
### X-Risk
- **AUROC**: [Optimizing AUROC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/02_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb)
- **AUPRC**: [Optimizing AUPRC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/03_Optimizing_AUPRC_Loss_on_Imbalanced_dataset.ipynb)
- **Partial AUROC**: [Optimizing Partial AUC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/11_Optimizing_pAUC_Loss_on_Imbalanced_data_wrapper.ipynb)
- **Compositional AUROC**: [Optimizing Compositional AUROC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/09_Optimizing_CompositionalAUC_Loss_with_ResNet20_on_CIFAR10.ipynb)
- **NDCG**: [Optimizing NDCG loss on MovieLens 20M](https://github.com/Optimization-AI/LibAUC/blob/main/examples/10_Optimizing_NDCG_Loss_on_MovieLens20M.ipynb) 
- **SogCLR**: [Optimizing Contrastive Loss using small batch size on ImageNet-1K](https://github.com/Optimization-AI/SogCLR)

### Applications
- [A Tutorial of Imbalanced Data Sampler](https://github.com/Optimization-AI/LibAUC/blob/main/examples/placeholder.md) (Updates Coming Soon)
- [Constructing benchmark imbalanced datasets for CIFAR10, CIFAR100, CATvsDOG, STL10](https://github.com/Optimization-AI/LibAUC/blob/main/examples/01_Creating_Imbalanced_Benchmark_Datasets.ipynb)
- [Using LibAUC with PyTorch learning rate scheduler](https://github.com/Optimization-AI/LibAUC/blob/main/examples/04_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb)
- [Optimizing AUROC loss on Chest X-Ray dataset (CheXpert)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)
- [Optimizing AUROC loss on Skin Cancer dataset (Melanoma)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/08_Optimizing_AUROC_Loss_with_DenseNet121_on_Melanoma.ipynb)
- [Optimizing AUROC loss on Molecular Graph dataset (OGB-Molhiv)](https://github.com/yzhuoning/DeepAUC_OGB_Challenge)
- [Optimizing multi-label AUROC loss on Chest X-Ray dataset (CheXpert)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/07_Optimizing_Multi_Label_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)
- [Optimizing AUROC loss on Tabular dataset (Credit Fraud)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/placeholder.md) (Updates Coming Soon) 
- [Optimizing AUROC loss for Federated Learning](https://github.com/Optimization-AI/LibAUC/blob/main/examples/scripts/06_Optimizing_AUROC_loss_with_DenseNet121_on_CIFAR100_in_Federated_Setting_CODASCA.py)


:page_with_curl: Citation
---------
If you find LibAUC useful in your work, please cite the papers in [BibTex](https://github.com/Optimization-AI/LibAUC/blob/main/citations.bib) and acknowledge our library:
```
@software{libauc2022,
  title={LibAUC: A Deep Learning Library for X-risk Optimization.},
  author={Yuan, Zhuoning and Qiu, Zi-Hao and Li, Gang and Zhu, Dixian and Guo, Zhishuai and Hu, Quanqi and Wang, Bokun and Qi, Qi and Zhong, Yongjian and Yang, Tianbao },
  year={2022}
  }
 ```
 ```
@article{yang2022algorithmic,
  title={Algorithmic Foundation of Deep X-Risk Optimization},
  author={Yang, Tianbao},
  journal={arXiv preprint arXiv:2206.00439},
  year={2022}
}
```

:email: Contact
----------
For any technical questions, please open a new issue in the Github. If you have any other questions, please contact us @ [Zhuoning Yuan](https://zhuoning.cc) [yzhuoning@gmail.com] and [Tianbao Yang](http://people.tamu.edu/~tianbao-yang/) [tianbao-yang@tamu.edu]. 
