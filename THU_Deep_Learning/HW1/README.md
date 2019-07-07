# MNIST Digits Classification with MLP
This project uses MLP to perform MNIST classification.


## Getting Started
The zip folder contains a Report.pdf, codes folder, and the instruction pdf.

To reproduce results shown in Report.pdf, user can run *run_mlp.py* in *codes* folder.

To try out other model/loss function/hyperparameters, user should toggle the respective parts in *run_mlp.py*.

## Changes to Original *codes*

* ### *load_data.py*
*unzip_mnist_2d()* is added to unzip .gz files.

* ### *utils.py*
*make_plot()* is added as a helper function used in *run_mlp.py* to plot charts.

* ### *run_mlp.py*
*run_mlp.py* now includes a nested loop to run over the models and losses.


## Author
* **Zhang Naifu** - *2018280351*

## References
* [Deep Learning](http://www.deeplearningbook.org/)
* [CS231](http://cs231n.github.io/neural-networks-case-study/)
