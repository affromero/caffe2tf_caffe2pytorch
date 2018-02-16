#!/usr/bin/env python
import os
import numpy as np
import argparse
import caffe

caffe.set_mode_gpu()

def convert(def_path, caffemodel, data_output_path, phase):
  net = caffe.Net(def_path, caffemodel, getattr(caffe, phase.upper()))
  weights = net.params
  layers = weights.keys()
  npy_data = {}
  wb = [0,1]
  wb_new = ['weights', 'biases']
  for l in layers:
    npy_data[l] = {}
    for i,j in zip(wb, wb_new):
      npy_data[l][j] = weights[l][i].data
      if i==0: print("Saving "+l+", Weigths: "+str(weights[l][i].data.shape))
      elif i==1: print("Saving "+l+", Biases: "+str(weights[l][i].data.shape))
  np.save(data_output_path, npy_data)


#TO LOAD
#data_dict = np.load(npy_path, encoding='latin1').item()