# MNIST Digits Classification with PyTorch
This project uses PyTorch to build MLP & CNN to perform MNIST classification.


## Getting Started
The zip folder contains a Report.pdf, codes folder, and the instruction pdf.

To reproduce results shown in Report.pdf, user can run *run_cnn.py* & *run_mlp.py* in *codes* folder.

To try out other hyperparameters, user should toggle the respective parts in *run_cnn.py* & *run_mlp.py*.

## Changes to Original *codes*

* ### *utils.py*
*plot_loss_acc* is added as a helper function used to plot loss & acc charts.

* ### *run_cnn.py* & *run_mlp.py*
Minimum changes made here to return some intermediate results.


## Author
* **Zhang Naifu** - *2018280351*

## References
* [Deep Learning](http://www.deeplearningbook.org/)
* [CS231n](http://cs231n.github.io/neural-networks-2/)
* [Ioffe, S., Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
