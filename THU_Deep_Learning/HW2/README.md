# MNIST Digits Classification with CNN
This project uses CNN to perform MNIST classification.


## Getting Started
The zip folder contains a Report.pdf, codes folder, and the instruction pdf.

To reproduce results shown in Report.pdf, user can run *run_cnn.py* in *codes* folder.

To try out other hyperparameters, user should toggle the respective parts in *run_cnn.py*.

## Changes to Original *codes*

* ### *load_data.py*
*unzip_mnist_2d()* is added to unzip .gz files.

* ### *utils.py*
*make_plot()* is added as a helper function used in *run_cnn.py* to plot loss & acc charts.

 *vis_square()* is added as a helper function to plot output after convolution.

* ### *run_cnn.py*
In addition to the functions to complete, *run_cnn.py* also contains 4 helper functions:

 3 functions for *im2col* and *col2im*;

 *make_W_ave()* for creating kernel matrix of the average pooling layer.

* ### *network.py* & *solve_net.py*
Minimum changes made here to return some intermediate results.


## Author
* **Zhang Naifu** - *2018280351*

## References
* [Deep Learning](http://www.deeplearningbook.org/)
* [CS231n](http://cs231n.github.io/convolutional-networks/)
* [Agustinus Kristiadi's Blog](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/)
* [Caffe tutorial](https://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb)
