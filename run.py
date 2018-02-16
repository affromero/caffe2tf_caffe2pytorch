#!/usr/bin/env python
import numpy as np
import argparse  
import os
import sys
sys.path.append('/home/afromero/caffe-master/python')
os.environ['GLOG_minloglevel'] = '3'

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--prototxt', help='Model definition (.prototxt) path')
  parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
  parser.add_argument('--npymodel', default=None, help='Converted data output path')
  parser.add_argument('--mode', required=True, choices=('tf', 'pytorch'), help='Converted data output path')
  args = parser.parse_args()

  if args.mode == 'tf':
  	from caffe_tensorflow.caffe2npy import convert
  else:
  	from caffe_pytorch.caffe2npy import convert

  if args.npymodel is None:
  	args.npymodel = args.caffemodel.replace('.caffemodel', '_'+args.mode+'.npy')
  print(args)
  convert(args.prototxt, args.caffemodel, args.npymodel, 'test')
  #globals().update(vars(args))
  print("succesfully saved at %s"%(args.npymodel))


if __name__ == '__main__':
    main()

#USAGE
# ./run.py --prototxt deploy.pt --caffemodel model.caffemodel --mode pytorch

# In order to load: 
# data_dict = np.load(npymodel, encoding='latin1').item()