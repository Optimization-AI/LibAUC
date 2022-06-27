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

:mag: What is X-Risks?
---
X-risk refers to a family of compositional measures/losses, in which each data point is compared with a set of data points explicitly or implicitly for defining a risk function. It covers a family of widely used measures/losses, which can be organized into four interconnected categories:
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
$ pip install libauc==1.1.9rc3
```
The latest version `1.1.9rc3` is available now! You can also download source code for previous version [here](https://github.com/Optimization-AI/LibAUC/releases). 


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
- **AUPRC**: [Optimizing AUPRC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/03_Optimizing_AUPRC_with_ResNet18_on_Imbalanced_CIFAR10.ipynb)
- **Partial AUROC**: [Optimizing Partial AUC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/placeholder.md) (Available soon)
- **Compositional AUROC**: [Optimizing Compositional AUROC loss on imbalanced dataset](https://github.com/Optimization-AI/LibAUC/blob/main/examples/09_Optimizing_CompositionalAUC_Loss_with_ResNet20_on_CIFAR10.ipynb)
- **NDCG**: [Optimizing NDCG loss on MovieLens 20M](https://github.com/Optimization-AI/LibAUC/blob/main/examples/10_Optimizing_NDCG_Loss_on_MovieLens20M.ipynb) 
- **SogCLR**: [Optimizing Contrastive Loss using small batch size on ImageNet-1K](https://github.com/Optimization-AI/SogCLR)

### Applications
- [A Tutorial of Imbalanced Data Sampler](https://github.com/Optimization-AI/LibAUC/blob/main/examples/placeholder.md) (Available soon)
- [Constructing benchmark imbalanced datasets for CIFAR10, CIFAR100, CATvsDOG, STL10](https://github.com/Optimization-AI/LibAUC/blob/main/examples/01_Creating_Imbalanced_Benchmark_Datasets.ipynb)
- [Using LibAUC with PyTorch learning rate scheduler](https://github.com/Optimization-AI/LibAUC/blob/main/examples/04_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb)
- [Optimizing AUROC loss on Chest X-Ray dataset (CheXpert)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)
- [Optimizing AUROC loss on Skin Cancer dataset (Melanoma)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/08_Optimizing_AUROC_Loss_with_DenseNet121_on_Melanoma.ipynb)
- [Optimizing AUROC loss on Molecular Graph dataset (OGB-Molhiv)](https://github.com/yzhuoning/DeepAUC_OGB_Challenge)
- [Optimizing multi-task AUROC loss on Chest X-Ray dataset (CheXpert)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/07_Optimizing_Multi_Label_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)
- [Optimizing AUROC loss on Tabular dataset (Credit Fraud)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/placeholder.md)(Available soon) 
- [Optimizing AUROC loss for Federated Learning](https://github.com/Optimization-AI/LibAUC/blob/main/examples/scripts/06_Optimizing_AUROC_loss_with_DenseNet121_on_CIFAR100_in_Federated_Setting_CODASCA.py)


:page_with_curl: Citation
---------
If you find LibAUC useful in your work, please acknowledge our library and cite the following papers:
```
@misc{libauc2022,
      title={LibAUC: A Deep Learning Library for X-Risk Optimization.},
      author={Zhuoning Yuan, Zi-Hao Qiu, Gang Li, Dixian Zhu, Zhishuai Guo, Quanqi Hu, Bokun Wang, Qi Qi, Yongjian Zhong, Tianbao Yang},
      year={2022}
	}
```
```
@article{dox22,
	title={Algorithmic Foundation of Deep X-risk Optimization},
	author={Tianbao Yang},
	journal={CoRR},
	year={2022}
```

:email: Contact
----------
For any technical questions, please open a new issue in the Github. If you have any other questions, please contact us @ [Zhuoning Yuan](https://zhuoning.cc) [yzhuoning@gmail.com] and [Tianbao Yang](https://homepage.cs.uiowa.edu/~tyng/) [tianbao-yang@uiowa.edu]. 
