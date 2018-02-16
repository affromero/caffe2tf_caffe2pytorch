# Caffe to TensorFlow | Caffe to Pytorch

Convert [Caffe](https://github.com/BVLC/caffe/) models to [TensorFlow](https://github.com/tensorflow/tensorflow) and [PyTorch](https://github.com/pytorch/pytorch).

## Caffe to Tensorflow 
This implementation is based on [ethereon's](https://github.com/ethereon/caffe-tensorflow) with minor changes. 

## Caffe to PyTorch
This implementation only extract caffemodel parameters and save them in a npy standard format. 

### Usage

Run `run.py` to convert an existing Caffe model to TensorFlow/PyTorch.

`./run.py --prototxt deploy.pt --caffemodel model.caffemodel --mode pytorch`

`./run.py --prototxt deploy.pt --caffemodel model.caffemodel --mode tf`

Make sure you're using the latest Caffe format (see the notes section for more info).

The output in the `tf` mode consists of two files:

1. A data file (in NumPy's native format) containing the model's learned parameters.
2. A Python class that constructs the model's graph.

Meanwhile, for the `pytorch` way only is the former file. 

### NPY output

The npy output consits of a dictionary with the corresponding layers. Each layer is itself a dictionary with `weights` and `biases`.
