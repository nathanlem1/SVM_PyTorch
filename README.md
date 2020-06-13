This repository implements a linear Support Vector Machine (SVM) using [PyTorch](https://github.com/pytorch/pytorch). The linear SVM can be implemented using fully connected layer and multi-class
classification hinge loss in PyTorch. We also include a logistic regression which uses cross-entropy loss which internally
computes softmax. In this implementation, we also include regularization techniques such as L1 (LASSO - Least Absolute Shrinkage and Selection Operator), 
L2 (Ridge) and Elastic Net (combination of L1 and L2) based on [Lecture note](http://cs231n.stanford.edu/slides/2020/lecture_3.pdf). 

We think this repository can be used as a complementary to the [pytorch-tutorial](https://github.com/nathanlem1/pytorch-tutorial) which implements
traditional supervised machine learning algorithms such as linear regression, logistic regression and feedforward neural 
network in addition to some advanced useful deep learning methods. In this repository, we implemented the linear SVM.  It is also recommended to look into the [Official Pytorch Tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
to start with if you are a beginner.


<br/>

## Getting Started
```bash
$ git clone https://github.com/nathanlem1/SVM_PyTorch.git
$ cd SVM_PyTorch
$ python SVM_PyTorch.py
```

<br/>

## Dependencies
* [Python 3.5+](https://www.python.org/downloads/)
* [PyTorch 0.4.0+](http://pytorch.org/)




