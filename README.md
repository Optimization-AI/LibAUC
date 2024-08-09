<p align="center">
  <img src="https://docs.libauc.org/_images/libauc_new_logo_v5.png" width="70%" align="center"/><br>
</p>



LibAUC: A Deep Learning Library for X-Risk Optimization
---
<p align="left">
  <a href="https://github.com/Optimization-AI/LibAUC">
    <img alt="Pypi" src="https://img.shields.io/pypi/v/libauc?color=blue&style=flat"/>
  </a>
  <a href="https://pepy.tech/project/libauc">
    <img alt="Downloads" src="https://static.pepy.tech/badge/LibAUC"/>
  </a>
  <a href="https://github.com/Optimization-AI/LibAUC">
    <img alt="python" src="https://img.shields.io/pypi/pyversions/libauc"/>
  </a>
  <a href="https://github.com/Optimization-AI/LibAUC">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0-yellow?color=blue&style=flat"/>
  </a>
  <a href="https://github.com/Optimization-AI/LibAUC/blob/main/LICENSE">
    <img alt="LICENSE" src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
  </a>
</p>

| [**Documentation**](https://docs.libauc.org/)
| [**Installation**](https://libauc.org/installation/)
| [**Website**](https://libauc.org/)
| [**Tutorial**](https://github.com/Optimization-AI/LibAUC/tree/main/examples)
| [**Research**](https://libauc.org/publications/)
| [**Github**](https://github.com/Optimization-AI/LibAUC/) |


News
--- 
- [2024/04/07]: **Bugs fixed:** We fixed a bug in datasets/folder.py by returning a return_index to support SogCLR/iSogCLR for contrastive learning. Fixed incorrect communication with all_gather in GCLoss_v1 and set gamma to original value when u is not 0. None of these were in our experimental code of the paper. 
- [2024/02/11]: **A Bug fixed:** We fixed a bug in the calculation of AUCM loss and MultiLabelAUCM loss (the margin parameter is missed in the original calculation which might cause the loss to be negative). However, it does not affect the learning as the updates are not affected by this. Both the source code and pip install are updated. 
- [2023/06/10]: **LibAUC 1.3.0 is now available!** In this update, we have made improvements and introduced new features. We also release a new documentation website at [https://docs.libauc.org/](https://docs.libauc.org/). Please see the [release notes](https://github.com/Optimization-AI/LibAUC/releases) for details. 
- [2023/06/10]: We value your thoughts and feedback! Please consider filling out [this brief survey](https://forms.gle/oWNtjN9kLT51CMdf9) to guide our future developments. Thank you!

Why LibAUC?
---
LibAUC offers an easier way to directly optimize commonly-used performance measures and losses with user-friendly API. LibAUC has broad applications in AI for tackling many challenges, such as **Classification of Imbalanced Data (CID)**, **Learning to Rank (LTR)**, and **Contrastive Learning of Representation (CLR)**. LibAUC provides a unified framework to abstract the optimization of many compositional loss functions, including surrogate losses for AUROC, AUPRC/AP, and partial AUROC that are suitable for CID, surrogate losses for NDCG, top-K NDCG, and listwise losses that are used in LTR, and global contrastive losses for CLR. Here’s an overview:

<p align="center">
  <img src="https://docs.libauc.org/_images/dxo-overview-v4.png" width="65%" align="center"/>
</p>


Installation
--------------
Installing from pip
```
$ pip install -U libauc
```

Installing from source

```
$ git clone https://github.com/Optimization-AI/LibAUC.git
$ cd LibAUC
$ pip install .
```
For more details, please check the latest [release note](https://github.com/Optimization-AI/LibAUC/releases/).




Usage
---
#### Example training pipline for optimizing X-risk (e.g., AUROC) 
```python
>>> #import our loss and optimizer
>>> from libauc.losses import AUCMLoss 
>>> from libauc.optimizers import PESG 
>>> #pretraining your model through supervised learning or self-supervised learning
>>> #load a pretrained encoder and random initialize the last linear layer 
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

Tutorials
-------
### X-Risk Minimization

- **Optimizing AUCMLoss**: [[example]](https://docs.libauc.org/examples/auroc.html)
- **Optimizing APLoss**: [[example]](https://docs.libauc.org/examples/auprc.html)
- **Optimizing CompositionalAUCLoss**: [[example]](https://docs.libauc.org/examples/compauc.html)
- **Optimizing pAUCLoss**: [[example]](https://docs.libauc.org/examples/pauc.html)
- **Optimizing MIDAMLoss**: [[example]](https://docs.libauc.org/examples/MIDAM-att-tabular.html)
- **Optimizing NDCGLoss**: [[example]](https://docs.libauc.org/examples/ndcg.html) 
- **Optimizing GCLoss (Unimodal)**: [[example]](https://docs.libauc.org/examples/sogclr.html)
- **Optimizing GCLoss (Bimodal)**: [[example]](https://docs.libauc.org/examples/isogclr.html)

<details markdown="1">
  <summary>Other Applications</summary>
  
  - [Constructing benchmark imbalanced datasets for CIFAR10, CIFAR100, CATvsDOG, STL10](https://github.com/Optimization-AI/LibAUC/blob/main/examples/01_Creating_Imbalanced_Benchmark_Datasets.ipynb)
  - [Using LibAUC with PyTorch learning rate scheduler](https://github.com/Optimization-AI/LibAUC/blob/main/examples/04_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb) 
  - [Optimizing AUROC loss on Chest X-Ray dataset (CheXpert)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)
  - [Optimizing AUROC loss on Skin Cancer dataset (Melanoma)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/08_Optimizing_AUROC_Loss_with_DenseNet121_on_Melanoma.ipynb)
  - [Optimizing multi-label AUROC loss on Chest X-Ray dataset (CheXpert)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/07_Optimizing_Multi_Label_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)
  - [Optimizing AUROC loss on Tabular dataset (Credit Fraud)](https://github.com/Optimization-AI/LibAUC/blob/main/examples/12_Optimizing_AUROC_Loss_on_Tabular_Data.ipynb)
  - [Optimizing AUROC loss for Federated Learning](https://github.com/Optimization-AI/LibAUC/blob/main/examples/scripts/06_Optimizing_AUROC_loss_with_DenseNet121_on_CIFAR100_in_Federated_Setting_CODASCA.py)
  - [Optimizing GCLoss (Bimodal with Cosine Gamma)](https://docs.libauc.org/examples/sogclr_gamma.html)
	
</details>


Citation
---------
If you find LibAUC useful in your work, please cite the following papers:
```
@inproceedings{yuan2023libauc,
	title={LibAUC: A Deep Learning Library for X-Risk Optimization},
	author={Zhuoning Yuan and Dixian Zhu and Zi-Hao Qiu and Gang Li and Xuanhui Wang and Tianbao Yang},
	booktitle={29th SIGKDD Conference on Knowledge Discovery and Data Mining},
	year={2023}
	}
 ```
 ```
@article{yang2022algorithmic,
	title={Algorithmic Foundations of Empirical X-Risk Minimization},
	author={Yang, Tianbao},
	journal={arXiv preprint arXiv:2206.00439},
	year={2022}
}
```

Contact
----------
For any technical questions, please open a new issue in the Github. If you have any other questions, please contact us via libaucx@gmail.com  or tianbao-yang@tamu.edu. 
