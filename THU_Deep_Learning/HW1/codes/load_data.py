import numpy as np
import os
from os import listdir
import gzip
import shutil

def unzip_mnist_2d(data_dir):
    files = listdir(data_dir)
    for filename in files:
        
        # check if filename is a gz file
        if filename[-3:] != '.gz':
            continue
        
        f_in_name = os.path.join(data_dir, filename)
        f_out_name = f_in_name[:-3]
            
        with gzip.open(f_in_name, 'rb') as f_in:
            with open(f_out_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                print(f_in_name + ' unzipped...')


def load_mnist_2d(data_dir):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = (trX - 128.0) / 255.0
    teX = (teX - 128.0) / 255.0

    return trX, teX, trY, teY


#unzip_mnist_2d('data')
