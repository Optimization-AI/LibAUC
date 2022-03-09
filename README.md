<p align="center">
  <img src="https://github.com/yzhuoning/LibAUC/blob/main/imgs/libauc.png" width="70%" align="center"/>
</p>
<p align="center">
  Logo by <a href="https://homepage.divms.uiowa.edu/~zhuoning/">Zhuoning Yuan</a>
</p>

**LibAUC**: A Machine Learning Library for AUC Optimization
---
<p align="left">
  <img alt="PyPI version" src="https://img.shields.io/pypi/v/libauc?color=blue&style=flat-square"/>
  <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/libauc?color=blue&style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.8-yellow?color=blue&style=flat-square" />	
  <img alt="PyPI LICENSE" src="https://img.shields.io/github/license/yzhuoning/libauc?color=blue&logo=libauc&style=flat-square" />
</p>

[**Website**](https://libauc.org/)
| [**Updates**](https://libauc.org/news/)
| [**Installation**](https://libauc.org/get-started/)
| [**Tutorial**](https://github.com/Optimization-AI/LibAUC/tree/main/examples)
| [**Research**](https://libauc.org/publications/)
| [**Github**](https://github.com/Optimization-AI/LibAUC/)

**LibAUC** aims to provide efficient solutions for optimizing AUC scores (*AUROC, AUPRC*). We will continuously update our library by fixing bugs and adding new features. If you use or like our library, please **star**:star: our repo. Thank you!



:mag: Why LibAUC?
---
**Deep AUC Maximization (DAM)** is a paradigm for learning a deep neural network by maximizing the AUC score of the model on a dataset. In practice, many real-world datasets are usually imbalanced and AUC score is a better metric for evaluating and comparing different methods. Directly maximizing AUC score can potentially lead to the largest improvement in the model’s performance since maximizing AUC aims to rank the prediction  score of any positive data higher than any negative data. Our library can be used in many applications, such as medical image classification and drug discovery.


:star:Key Features
---
- **[Easy Installation](https://github.com/Optimization-AI/LibAUC#key-features)** - Integrate *AUROC*, *AUPRC* training code with your existing pipeline in just a few steps
- **[Large-scale Learning](https://github.com/Optimization-AI/LibAUC#key-features)** - Handle large-scale optimization and make the training more smoothly
- **[Distributed Training](https://github.com/Optimization-AI/LibAUC#key-features)** - Extend to distributed setting to accelerate training efficiency and enhance data privacy
- **[ML Benchmarks](https://github.com/Optimization-AI/LibAUC#key-features)** - Provide easy-to-use input pipeline and benchmarks on various datasets


:gear: Installation
--------------
```
$ pip install libauc
```
You can also download source code from [here](https://github.com/Optimization-AI/LibAUC/releases).

:notebook_with_decorative_cover: Usage
-------
### Official Tutorials:
- Constructing Imbalanced datasets for **CIFAR10, CIFAR100, CATvsDOG, STL10** [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/01_Creating_Imbalanced_Benchmark_Datasets.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Training with Pytorch Learning Rate Scheduling [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/04_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Optimizing <strong>AUROC</strong> loss with ResNet20 on Imbalanced CIFAR10 [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/02_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Optimizing <strong>AUPRC</strong> loss with ResNet18 on Imbalanced CIFAR10 [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/03_Optimizing_AUPRC_with_ResNet18_on_Imbalanced_CIFAR10.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Optimizing <strong>AUROC</strong> loss with DenseNet121 on <strong>CheXpert</strong> [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/05_Optimizing_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Optimizing <strong>AUROC</strong> loss with DenseNet121 for **Federated Learning** [[Preliminary Release](https://github.com/Optimization-AI/LibAUC/blob/main/examples/scripts/06_Optimizing_AUROC_loss_with_DenseNet121_on_CIFAR100_in_Federated_Setting_CODASCA.py)]
- Optimizing <strong>AUROC</strong> loss with DenseNet121 on <strong>Melanoma</strong> [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/08_Optimizing_AUROC_Loss_with_DenseNet121_on_Melanoma.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Optimizing <strong>AUROC (Multi-Label)</strong> loss with DenseNet121 on <strong>CheXpert</strong> [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/07_Optimizing_Multi_Label_AUROC_Loss_with_DenseNet121_on_CheXpert.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]
- Optimizing <strong>AUROC</strong> loss with ResNet20 for <strong>Compositional Training</strong> [[Notebook](https://github.com/Optimization-AI/LibAUC/blob/main/examples/09_Optimizing_CompositionalAUC_Loss_with_ResNet20_on_CIFAR10.ipynb)][[Script](https://github.com/Optimization-AI/LibAUC/tree/main/examples/scripts)]



### Quickstart for Beginners:
#### Optimizing AUROC (Area Under the Receiver Operating Characteristic)
```python
>>> #import library
>>> from libauc.losses import AUCMLoss
>>> from libauc.optimizers import PESG
...
>>> #define loss
>>> Loss = AUCMLoss(imratio=[YOUR NUMBER])
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
>>> #restart stage
>>> optimizer.update_regularizer()
```


#### Optimizing AUPRC (Area Under the Precision-Recall Curve)
```python
>>> #import library
>>> from libauc.losses import APLoss
>>> from libauc.optimizers import SOAP
...
>>> #define loss
>>> Loss = APLoss()
>>> optimizer = SOAP()
...
>>> #training
>>> model.train()    
>>> for index, data, targets in trainloader:
>>>	data, targets  = data.cuda(), targets.cuda()
        logits = model(data)
	preds = torch.sigmoid(logits)
        loss = Loss(preds, targets, index) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()	

```

:zap: Useful Tips
---
- Your dataset should have **0,1** labels, e.g., **1** is the **minority class** and **0** is the **majority class**
- Compute `imratio=#pos/#total` based on training set and pass it to `AUCMLoss(imratio=xxx)`
- Adopt a proper `initial learning rate`, e.g., `lr=[0.1, 0.05]` usually works better
- Choose `libauc.optimizers.PESG` to optimize `AUCMLoss(imratio=xxx)`
- Use `optimizer.update_regularizer(decay_factor=10)` to update learning rate and regularizer in stagewise
- Add activation layer, e.g., `torch.sigmoid(logits)`, before passing model outputs to loss function 
- Reshape both variables `y_preds` and `y_targets` to `(N, 1)` before calling loss function


:page_with_curl: Citation
---------
If you find LibAUC useful in your work, please acknowledge our library and cite the following paper:
```
@inproceedings{yuan2021robust,
	title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
	author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	year={2021}
	}
```

:email: Contact
----------
If you have any questions, please contact us @ [Zhuoning Yuan](https://homepage.divms.uiowa.edu/~zhuoning/) [yzhuoning@gmail.com] and [Tianbao Yang](https://homepage.cs.uiowa.edu/~tyng/) [tianbao-yang@uiowa.edu] or please open a new issue in the Github . 
