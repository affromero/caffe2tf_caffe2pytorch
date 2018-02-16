#!/usr/bin/env python
import numpy as np
import os
import argparse
from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer

def fatal_error(msg):
    print_stderr(msg)
    exit(-1)

def convert(def_path, caffemodel_path, data_output_path, phase):
    assert '.npy' in data_output_path
    code_output_path = data_output_path.replace('npy', 'py')
    try:
        transformer = TensorFlowTransformer(def_path, caffemodel_path, phase=phase)
        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            # create directory if not existing
            dirname = os.path.dirname(data_output_path)
            if not os.path.exists(dirname) and dirname != '':
                os.makedirs(dirname)
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            # create directory if not existing
            dirname = os.path.dirname(code_output_path)
            if not os.path.exists(dirname) and dirname != '':
                os.makedirs(dirname)
            with open(code_output_path, 'wb') as src_out:
                src_out.write(transformer.transform_source())
        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))
